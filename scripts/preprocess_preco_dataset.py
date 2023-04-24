import jsonlines
import numpy as np
import matplotlib.pyplot as plt
import json

def preprocess(split):
    with jsonlines.open("/mnt/ssd/coreference_resolution_evaluation/test/"+split+".jsonl", "r") as reader:
        with jsonlines.open("/mnt/ssd/coreference_resolution_evaluation/test/"+split+"_preprocessed_preco.jsonlines", "w") as writer:
            for obj in reader:
                res = {}
                res["doc_key"] = obj["id"]
                lens = [len(x) for x in obj["sentences"]]
                res["sentences"] = obj["sentences"]
                coreferences = []
                for cluster in obj["mention_clusters"]:
                    clusters = []
                    if len(cluster) == 1:
                        continue
                    for span in cluster:
                        offset = span[0]
                        start = sum(lens[:offset]) + span[1]
                        end = sum(lens[:offset]) + span[2]-1
                        clusters.append([start,end])
                    if cluster != []:
                        coreferences.append(clusters)
                res["clusters"] = coreferences
                writer.write(res)