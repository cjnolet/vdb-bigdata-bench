from pymilvus import CollectionSchema, FieldSchema, DataType, utility, connections, Collection, list_collections

import time
import struct
import numpy as np
import os
import random
import json
import time
import os

collection_name = "VectorDBBenchCollection"
connections.connect(host="localhost", port=19530)
dim = 1024
print(f"\nList collections...", flush=True)
collection_list = list_collections()
print(list_collections())

coll = Collection(collection_name)
coll.release()
coll.drop_index()

print("create index", flush=True)
s = time.time()
try:
    coll.create_index(field_name="base",
        index_params={'index_type': 'GPU_IVF_PQ',
            'metric_type': 'L2',
            'params': {
                'nlist': 50000,
                'm': 16
                }})
except Exception as e:
    print(f"index error: {e}", flush=True)
    raise e from None

index_t  = time.time()
print("create index time :", index_t - s, flush=True)

print("load index")
coll.load()
load_t = time.time()
print("load index time :", load_t - index_t, flush=True)
