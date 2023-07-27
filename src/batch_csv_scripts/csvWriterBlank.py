import os
import csv
from tqdm import tqdm

root = "C:\\Users\\samue\\OneDrive - University of Cambridge\\Summer Research Project\\data"

folder = os.path.join(root, "batch2", "orig", "jpg")
csvPath = os.path.join(root, "batch2", "data_new.csv")

files = os.listdir(folder)

# print(files)

with open(csvPath, "w", encoding='UTF8', newline="") as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=",")
    csv_writer.writerow([
        "filename", "location", "side", "area", "major", "minor", "circ.",
        "feret", "minferet", "ar"
    ] + ["efd_" + str(i) for i in range(25)] + [
        "colour_" + str(i) + "_" + str(j) for i in range(4)
        for j in ["r", "g", "b"]
    ] + ["percentage_" + str(i) for i in range(4)] + ["species"])

    for file in tqdm(files):
        csv_writer.writerow([file, "", "", "", "", "", "", "", "", ""] +
                            [""] * 25 + [""] * 12 + [""])