import os
import csv
import cv2
from imageJ_funcs.analyse import Batch, analyse_as_string
from efd import efd_as_string
from colours.kmeans import get_kmeans_as_string
from tqdm import tqdm

dataRoot = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                        "split_data", "train")

oldCSVFilepath = os.path.join(dataRoot, "train_data_formatted_all.csv")
csvFilepath = os.path.join(dataRoot, "new_results_2.csv")

batch1, batch2 = [
    set(
        map(lambda x: x[:-6],
            os.listdir(os.path.join(dataRoot, "images", f"batch{i}", "orig"))))
    for i in range(1, 3)
]

batch_map = {Batch.ONE: "batch1", Batch.TWO: "batch2"}
location_map = ["front", "rear"]
wing_map = ["left", "right"]
side_map = ["_v", "_d"]

def find_batch(file):
    if file[:-6] in batch1:
        return Batch.ONE
    if file[:-6] in batch2:
        return Batch.TWO
    return None

with open(oldCSVFilepath, 'r', newline='', encoding='UTF8') as oldCSVFile:
    with open(csvFilepath, 'w', newline='', encoding='UTF8') as csvFile:

        writer = csv.writer(csvFile)
        headers = [
            "filename", "location", "wing", "side", "area", "major", "minor",
            "circ", "feret", "minferet", "ar"
        ] + [f"efd_{i}" for i in range(25)] + [
            f"colour_{i}_{j}" for i in range(2) for j in ["r", "g", "b"]
        ] + [f"percentage_{i}" for i in range(2)] + ["species"]

        writer.writerow(headers)

        oldCSVReader = csv.reader(oldCSVFile)

        next(oldCSVReader)  # skip header row

        for row in tqdm(list(oldCSVReader)):
            filename = row[0]
            location = row[1]
            wing = row[2]
            side = row[3]
            species = row[-1]
            batch = find_batch(filename)

            if batch is None:
                raise KeyError(f"Batch not found for {filename}")

            fileRoute = os.path.join(location_map[int(location)],
                                     wing_map[int(wing)], filename)

            filepath = os.path.join(dataRoot, "images", batch_map[batch],
                                    "split_fixed", fileRoute)

            if not os.path.exists(filepath):
                print(f"Image {filepath} does not exist")
                continue

            area, major, minor, circ, maxFeret, minFeret, ar = analyse_as_string(
                filepath, BATCH=batch, print_results=False)

            efdCoeffs, efd_image = efd_as_string(filepath)

            kmeans, kmeans_image = get_kmeans_as_string(filepath, n_clusters=2)

            kmeansWriteResult = cv2.imwrite(
                os.path.join(dataRoot, "images", batch_map[batch],
                             "processed_2", "kmeans", fileRoute), kmeans_image)

            efdWriteResult = cv2.imwrite(
                os.path.join(dataRoot, "images", batch_map[batch],
                             "processed_2", "efd", fileRoute), efd_image)

            if not kmeansWriteResult:
                raise IOError(f"Could not write k-means image for {filename}")

            if not efdWriteResult:
                raise IOError(f"Could not write EFD image for {filename}")

            row = [
                filename, location, wing, side, area, major, minor, circ,
                maxFeret, minFeret, ar
            ] + efdCoeffs + kmeans + [species]

            writer.writerow(row)
