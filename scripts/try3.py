import jsonlines
import numpy as np
import matplotlib.pyplot as plt

#with jsonlines.open("/mnt/ssd/coreference_resolution_evaluation/inputs/test.english.jsonlines") as reader:
with jsonlines.open("/mnt/ssd/nlp-template/data/prepare_ontonotes/dev.english.jsonlines") as reader:
    distances = []
    for  obj in reader:
        text = [item for sublist in obj["sentences"] for item in sublist]
        for cluster in obj["clusters"]:
            for span in cluster:
                distances.append(span[1]-span[0])

distances = np.array(distances)
plt.hist(distances, density=True, bins=62)
plt.show()

