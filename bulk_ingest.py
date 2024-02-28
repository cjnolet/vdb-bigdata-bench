from pymilvus import CollectionSchema, FieldSchema, DataType, utility, connections, Collection, list_collections

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from cuml.dask.datasets import make_blobs
from dask.array import to_npy_stack

import time
import struct
import numpy as np
import os
import random
import json
import time
import os
import shutil

from minio import Minio
from minio.error import S3Error

import environs

collection_name = "VectorDBBenchCollection"

# minio
DEFAULT_BUCKET_NAME = "a-bucket"
MINIO_ADDRESS = "0.0.0.0:9000"
MINIO_SECRET_KEY = "minioadmin"
MINIO_ACCESS_KEY = "minioadmin"

if __name__ == "__main__":
    env = environs.Env()
    env.read_env(".env")
    dataset_name = env.str("DATASET", "blobs")
    print(dataset_name)
    data_minio_path = dataset_name + "_npy_data/"
    print(data_minio_path)

    # n_rows = 635780000
    n_rows = 50000000
    batch_size = 50000000
    dim = 1024
    n_centers = 1024
    n_chunks = 30
    n_batches = np.ceil(n_rows / batch_size)

    # cluster = LocalCUDACluster(threads_per_worker=1)
    # client = Client(cluster)
    # workers = list(client.scheduler_info()['workers'].keys())

    # for batch in range(int(n_batches)):
    #     adjusted_batch_size = min(batch_size, n_rows - batch * batch_size)
    #     if batch == 0:
    #         X, y, centers = make_blobs(adjusted_batch_size, dim, centers=n_centers, workers=workers, n_parts = n_chunks, return_centers = True)
    #     else:
    #         X, y = make_blobs(adjusted_batch_size, dim, centers=centers, workers=workers, n_parts = n_chunks)
    #     to_npy_stack(data_minio_path + '/' + str(batch), X, axis=0)

    # client.close()
    # cluster.close()

    connections.connect(host="localhost", port=19530)
    print("connected swuccess")

    def upload(data_folder: str,
            bucket_name: str=DEFAULT_BUCKET_NAME)->(bool, list):
        if not os.path.exists(data_folder):
            print("Data path '{}' doesn't exist".format(data_folder))
            return False, []

        remote_files = []
        try:
            print("Prepare upload files")
            minio_client = Minio(endpoint=MINIO_ADDRESS, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
            found = minio_client.bucket_exists(bucket_name)
            if not found:
                print("MinIO bucket '{}' doesn't exist".format(bucket_name))
                return False, []

            remote_data_path = "milvus_bulkinsert"
            def upload_files(folder:str):
                for parent, dirnames, filenames in os.walk(folder):
                    print(parent, dirnames, filenames)
                    if parent is folder:
                        chunk = 0
                        for filename in filenames:
                            ext = os.path.splitext(filename)
                            if len(ext) != 2 or (ext[1] != ".json" and ext[1] != ".npy"):
                                continue
                            local_full_path = os.path.join(parent, filename)
                            minio_file_path = os.path.join(remote_data_path, os.path.basename(folder), filename)
                            minio_client.fput_object(bucket_name, minio_file_path, local_full_path)
                            print("Upload file '{}' to '{}'".format(local_full_path, minio_file_path))
                            remote_files.append(minio_file_path)
                        for dir in dirnames:
                            upload_files(os.path.join(parent, dir))

            upload_files(data_folder)

        except S3Error as e:
            print("Failed to connect MinIO server {}, error: {}".format(MINIO_ADDRESS, e))
            return False, []

        print("Successfully upload files: {}".format(remote_files))
        return True, remote_files

    print(f"\nList collections...")
    collection_list = list_collections()
    print(list_collections())

    if(collection_list.count(collection_name)):
        print(collection_name, " exist, and drop it")
        collection = Collection(collection_name)
        collection.drop()
        print("drop")
    print("create collection")

    coll = Collection(collection_name)

    begin_t = time.time()
    ok, remote_files = upload(data_folder=data_minio_path)

    print("do_bulk_insert")
    task_id = utility.do_bulk_insert(
        collection_name=collection_name,
        files=remote_files)

    print("wait insert")
    while True:
        task = utility.get_bulk_insert_state(task_id=task_id)
        print("Task state:", task.state_name)

        if coll.num_entities == n_rows:
            coll.flush()
            break
        time.sleep(1)
    insert_t  = time.time()
    print("bulk insert time:", insert_t - begin_t)

    print("create index")
    try:
        coll.create_index(field_name="base",
            index_params={'index_type': 'GPU_CAGRA',  
                'metric_type': 'L2',
                'params': {
                    'intermediate_graph_degree':64,
                    'graph_degree': 32,
                    'M':14,
                    'efConstruction': 360,
                    "nlist":1024,
                    }})
    except Exception as e:
        print(f"index error: {e}")
        raise e from None

    def wait_index():
        while True:
            progress = utility.index_building_progress(collection_name)
            print(progress)
            if progress.get("pending_index_rows", -1) == 0:
                break
            time.sleep(5)
            
    print("wait index")
    wait_index()
    index_t  = time.time()
    print("create index time :", index_t - insert_t)

    print("load index")
    coll.load()
    load_t = time.time()
    print("load index time :", load_t - index_t)
    print("total time:", load_t - begin_t)