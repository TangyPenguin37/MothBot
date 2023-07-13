import os
from tqdm import tqdm

root = "C:\\Users\\samue\\OneDrive - University of Cambridge\\Summer Research Project\\data"

subsets = [
    os.path.join(root, "labelled", "orig", "arm"),
    os.path.join(root, "labelled", "orig", "zea"),
]

supersets = [
    os.path.join(root, "coll.raw.all.sorted", "jpg"),
    os.path.join(root, "unlabelled_batch1", "orig", "jpg")
]

supersetFiles = [os.listdir(superset) for superset in supersets]

for subset in subsets:
    subsetFiles = os.listdir(subset)
    for subsetFile in tqdm(subsetFiles):
        if subsetFile not in supersetFiles[
                0] and subsetFile not in supersetFiles[1]:
            print(subsetFile)