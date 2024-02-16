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
print(f"\nList collections...")
collection_list = list_collections()
print(list_collections())

if(collection_list.count(collection_name)):
    print(collection_name, " exist, and drop it")
    coll = Collection(collection_name)
    coll.drop()
    print("drop")

print("create collection")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, dim=dim),
    FieldSchema(name="base", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields)

coll = Collection(collection_name, schema)
print("Done creating collection", flush=True)

begin_t = time.time()

import cuml
import numpy as np
from multiprocessing import Process

pool_size = 10
batch_size = 10000
total_vecs = 10000000

vecs_per_process = total_vecs / pool_size
with cuml.using_output_type('numpy'):
    X, y = cuml.datasets.make_blobs(batch_size, dim)
X = X.astype("float").tolist()

s = time.time()
def insert_vecs(id):
  print("inserting items %s" % id, flush=True)
  total_vecs_per_process = 0
  while total_vecs_per_process < vecs_per_process:
      ids = np.arange(batch_size).astype("int64") * id + total_vecs_per_process
      data = [
          ids.tolist(),
          X
      ]
      mr = coll.insert(data)
      total_vecs_per_process = total_vecs_per_process + batch_size

processes = []
for i in range(pool_size):
    p = Process(target=insert_vecs, args=(i,))
    p.start()
    processes.append(p)

for i in processes:
    i.join()

print("Ingested %s vectors in %s seconds" % (total_vecs, time.time() - s), flush=True)

