import dataclasses
import os
import pathlib
import shutil
import time
from pprint import pprint
from typing import Iterator, Optional

import cupy as cp
import dask.array as da
import numpy as np
import pymilvus
from cuml.dask.datasets import make_blobs
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from minio import Minio

# Params
N_ROWS = int(os.getenv("N_ROWS", 100000))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 10000))
DIM = int(os.getenv("DIM", 1024))
SKIP_UPLOAD = bool(os.getenv("SKIP_UPLOAD", False))

MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_DATA_NODE_COUNT = 4  # This should be the same as `milvus.dataNodeCount` in the milvus custom values yaml

MINIO_ADDRESS = "0.0.0.0:9000"
MINIO_SECRET_KEY = "minioadmin"
MINIO_ACCESS_KEY = "minioadmin"
# This should be the same as `minio.bucketName` in the milvus custom values yaml
DEFAULT_BUCKET_NAME = "milvus-bucket"

REMOTE_DATA_PATH = "milvus_bulkinsert"
COLLECTION_NAME = "VectorDBBenchCollection"
LOCAL_DATA_PATH = (pathlib.Path() / "blobs_npy_data").resolve()
LOCAL_DATA_PATH.mkdir(parents=True, exist_ok=True)
print(LOCAL_DATA_PATH)

# FIXME: it only works if the column name matches the file name
DATA_COL_NAME = "data"


_minio_client = None


def get_minio_client():
    global _minio_client
    if _minio_client is None:
        _minio_client = Minio(
            endpoint=MINIO_ADDRESS,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False,
        )
    return _minio_client


def generate_data_files(
    n_rows: int, batch_size: int, dim: int, n_centers: int
) -> Iterator[pathlib.Path]:
    """
    Milvus data files should not exceed 16 GB.
    So a file with 3.5 M rows (=14 GB) is a good choice.
    That is, choose the batch size and n_parts such that (n_rows / (batch_size * n_parts)) <= 3.5M
    """
    n_batches = np.ceil(n_rows / batch_size)
    print("n_batches:", n_batches)
    cluster = LocalCUDACluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)
    try:
        workers = list(client.scheduler_info()["workers"].keys())
        for batch in range(int(n_batches)):
            adjusted_batch_size = min(batch_size, n_rows - batch * batch_size)
            if batch == 0:
                X, y, centers = make_blobs(
                    adjusted_batch_size,
                    dim,
                    centers=n_centers,
                    return_centers=True,
                    workers=workers,
                )
            else:
                X, y = make_blobs(
                    adjusted_batch_size, dim, centers=centers, workers=workers
                )
            # TODO: numpy stack causes Milvus to complain about "no corresponding field in collection".
            # This is likely due to schema column name mapping errors.
            # da.to_npy_stack(DATA_MINIO_PATH / str(batch), X, axis=0)
            data_path = LOCAL_DATA_PATH / str(batch) / "{}.npy".format(DATA_COL_NAME)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            cp.save(data_path, X.compute())
            yield data_path
    finally:
        client.close()
        cluster.close()


def upload_file(local_path, minio_path):
    minio_client = get_minio_client()
    minio_client.fput_object(
        bucket_name=DEFAULT_BUCKET_NAME,
        object_name=minio_path,
        file_path=local_path,
        part_size=100 * 1024 * 1024,
    )
    return minio_path


def initialize_upload():
    # Create the MinIO bucket if it doesn't exist
    minio_client = Minio(
        endpoint=MINIO_ADDRESS,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    found = minio_client.bucket_exists(DEFAULT_BUCKET_NAME)
    if not found:
        print(
            "MinIO bucket '{}' doesn't exist, creating...".format(DEFAULT_BUCKET_NAME)
        )
        minio_client.make_bucket(DEFAULT_BUCKET_NAME)

    # TODO: Should we delete the existing files locally and in minio?
    # Delete the local data directory if it exists
    if LOCAL_DATA_PATH.exists():
        print("Data file directory already exists locally. Deleting...")
        shutil.rmtree(LOCAL_DATA_PATH)

    # Delete the remote data files if they exist
    remote_data_objects = list(
        minio_client.list_objects(
            DEFAULT_BUCKET_NAME, prefix=REMOTE_DATA_PATH, recursive=True
        )
    )
    if remote_data_objects:
        print(
            "Data files already exist in MinIO. Deleting {} files...".format(
                len(remote_data_objects)
            )
        )
        for obj in remote_data_objects:
            minio_client.remove_object(DEFAULT_BUCKET_NAME, obj.object_name)


@dataclasses.dataclass
class Stats:
    n_rows: int
    batch_size: int
    dim: int
    generate_and_upload_time: Optional[float] = None
    bulk_insert_time: Optional[float] = None


def main():
    stats = Stats(n_rows=N_ROWS, batch_size=BATCH_SIZE, dim=DIM)

    if not SKIP_UPLOAD:
        initialize_upload()
        # Generate data files, then upload to MinIO
        n_centers = 1024

        generate_and_upload_t = time.time()
        for local_path in generate_data_files(
            n_rows=N_ROWS,
            batch_size=BATCH_SIZE,
            dim=DIM,
            n_centers=n_centers,
        ):
            minio_path = "{}/{}".format(
                REMOTE_DATA_PATH, local_path.relative_to(LOCAL_DATA_PATH)
            )
            print("Uploading data file {} to {}".format(local_path, minio_path))
            upload_file(local_path=local_path, minio_path=minio_path)
            local_path.unlink()  # TODO: delete the local file after uploading?

        stats.generate_and_upload_time = time.time() - generate_and_upload_t
        print("generate and upload time", stats.generate_and_upload_time)

    # Connect to Milvus and create a collection
    pymilvus.connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    print("Prepare Milvus collection")
    collection_list = pymilvus.list_collections()
    print(pymilvus.list_collections())
    if collection_list.count(COLLECTION_NAME):
        print(COLLECTION_NAME, " already exists. Dropping it.")
        pymilvus.drop_collection(COLLECTION_NAME)
        print("drop")
    print("create collection")
    fields = [
        pymilvus.FieldSchema(
            name="id", dtype=pymilvus.DataType.INT64, is_primary=True, auto_id=True
        ),
        pymilvus.FieldSchema(
            name=DATA_COL_NAME, dtype=pymilvus.DataType.FLOAT_VECTOR, dim=DIM
        ),
    ]
    schema = pymilvus.CollectionSchema(fields)
    coll = pymilvus.Collection(
        COLLECTION_NAME, schema, shards_num=MILVUS_DATA_NODE_COUNT
    )

    # Milvus bulk insert
    minio_client = get_minio_client()
    remote_data_files = [
        obj.object_name
        for obj in minio_client.list_objects(
            DEFAULT_BUCKET_NAME, prefix=REMOTE_DATA_PATH, recursive=True
        )
    ]
    insert_t = time.time()
    task_ids = [
        pymilvus.utility.do_bulk_insert(
            collection_name=COLLECTION_NAME, files=[remote_path]
        )
        for remote_path in remote_data_files
    ]
    remaining_tasks = set(task_ids)
    while remaining_tasks:
        print("Remaining tasks:", len(remaining_tasks))
        for tid in [*remaining_tasks]:
            task = pymilvus.utility.get_bulk_insert_state(task_id=tid)
            if task.state in (
                pymilvus.BulkInsertState.ImportFailed,
                pymilvus.BulkInsertState.ImportFailedAndCleaned,
            ):
                print("Task failed:")
                pprint(task)
                print("--------------------------------------------")
                remaining_tasks.remove(tid)
            elif task.state == pymilvus.BulkInsertState.ImportCompleted:
                print("Task completed:")
                pprint(task)
                print("--------------------------------------------")
                print("Total number of entities inserted:", coll.num_entities)
                coll.flush()
                remaining_tasks.remove(tid)
        time.sleep(5)

    stats.bulk_insert_time = time.time() - insert_t
    print("bulk insert time:", stats.bulk_insert_time)

    pprint(dataclasses.asdict(stats))

    if coll.num_entities == stats.n_rows:
        print("Successfully inserted all {} entities".format(coll.num_entities))
    else:
        print("Bulk insert failed.")
        print("Total number of entities inserted:", coll.num_entities)
        print("Expected number of entities:", stats.n_rows)


if __name__ == "__main__":
    main()
