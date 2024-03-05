from pymilvus import CollectionSchema, FieldSchema, DataType, utility, connections, Collection, list_collections

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from cuml.dask.datasets import make_blobs
from dask.array import transpose, to_npy_stack
import cuml
import time
import struct
import numpy as np
import os
import random
import json
import time
import os
import shutil

from dask import dataframe as dd
import dask.array as da
from minio import Minio
from minio.error import S3Error
import dask

import environs
import cupy
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
    data_minio_path = dataset_name + "_npy_data/"
    print(data_minio_path)

    n_rows = 35000000
    batch_size = 35000000
    dim = 1024
    n_centers = 1024
    n_parts = 10
    n_batches = np.ceil(n_rows / batch_size)
    
    remote_files = []

    def upload_chunk(x, batch_size, batch, block_id = None):
        if block_id is not None and x.size != 0:
            idx = str(batch * batch_size + block_id[0])
            remote_data_path = "milvus_bulkinsert"

            local_full_path = os.path.join(data_minio_path, idx + ".npy")
            cupy.save(local_full_path, x, allow_pickle=None)

            try:
                minio_client = Minio(endpoint=MINIO_ADDRESS, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
                found = minio_client.bucket_exists(DEFAULT_BUCKET_NAME)
                if not found:
                    print("MinIO bucket '{}' doesn't exist. Creating bucket".format(DEFAULT_BUCKET_NAME))
                    minio_client.make_bucket(DEFAULT_BUCKET_NAME)
            except S3Error as e:
                print("Failed to connect MinIO server {}, error: {}".format(MINIO_ADDRESS, e))

            minio_file_path = os.path.join(remote_data_path, idx, "base.npy")
            minio_client.fput_object(DEFAULT_BUCKET_NAME, minio_file_path, local_full_path)

            remote_files.append(minio_file_path)

        # Remove the temporary file
        # os.remove(local_full_path)
        return x
    
    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)
    workers = list(client.scheduler_info()['workers'].keys())

    for batch in range(int(n_batches)):
        print(batch)
        adjusted_batch_size = min(batch_size, n_rows - batch * batch_size)
        if batch == 0:
            X, y, centers = cuml.dask.datasets.make_blobs(adjusted_batch_size, dim, centers=n_centers, return_centers = True, workers = workers, n_parts = n_parts)
        else:
            X, y = cuml.dask.datasets.make_blobs(adjusted_batch_size, dim, centers=centers, workers = workers, n_parts = n_parts)
        
        # Iterate over array chunks and upload to MinIO
        dask.array.map_blocks(upload_chunk, X, batch_size, batch).compute()

    client.close()
    cluster.close()

    connections.connect(host="localhost", port=19530)

    print(f"\nList collections...")
    collection_list = list_collections()
    print(list_collections())

    if(collection_list.count(collection_name)):
        print(collection_name, " exist, and drop it")
        collection = Collection(collection_name)
        collection.drop()
        print("drop")
    print("create collection")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="base", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields)
    coll = Collection(collection_name, schema)

    print("remote files to be inserted", remote_files)

    begin_t = time.time()

    print("do_bulk_insert")
    for remote_file in remote_files:
        print(remote_file)
        task_id = utility.do_bulk_insert(
            collection_name=collection_name,
            files=[remote_file])

    print("wait insert")
    while True:
        tasks = utility.list_bulk_insert_tasks(collection_name=collection_name)
        for task in tasks:
            print(task)
        # task = utility.get_bulk_insert_state(task_id=task_id)
        # print("Task state:", task.state_name)
        # print("Imported files:", task.files)
        # print("Collection name:", task.collection_name)
        # print("Partition name:", task.partition_name)
        # print("Start time:", task.create_time_str)
        # print("Imported row count:", task.row_count)
        # print("Entities ID array generated by this task:", task.ids)
        # print("Task failed reason", task.failed_reason)

        print(coll.num_entities)

        if coll.num_entities >= n_rows:
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
