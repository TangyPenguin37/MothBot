import os
import numpy as np
from tqdm import trange
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

GROUPING_LEVEL = 1
EXCLUDED_COLUMNS = ["filename"] \
    + [
    f"{i}_d"
    for i in ["area", "major", "minor", "circ", "feret", "minferet", "ar"]
] + [f"efd_{i}_{j}" for i in range(25) for j in ["d", "v"]] + ["ar_v"]

suffixes = ["", "_grouped", "_grouped_further"]

filepath = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "data", "split_data",
    "train", "without_hybrids",
    f"new_results_no_hybrids{suffixes[GROUPING_LEVEL]}.csv")

with open(filepath, 'r', newline='', encoding='UTF8') as csvFile:
    headers = csvFile.readline().split(',')

columns = [
    i for i, header in enumerate(headers) if header not in EXCLUDED_COLUMNS
]

assert len(columns) == len(headers) - len(EXCLUDED_COLUMNS)

csvData = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=columns)

csvData = csvData[csvData[:, 0] == 1]
# csvData = csvData[csvData[:, 1] == 0]

x, y = np.split(csvData, [-1], 1)  # pylint: disable=unbalanced-tuple-unpacking
y = y.ravel()

def run(print_report=False):
    # split data into train and test sets
    train_x, test_x, train_y, test_y = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y)

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(train_x, train_y)

    if print_report:
        print(
            classification_report(test_y,
                                  lda.predict(test_x),
                                  target_names=["arm", "zea"],
                                  digits=3))

    return lda.score(test_x, test_y)

if __name__ == "__main__":
    accuracy = []
    for _ in trange(1000):
        accuracy.append(run())

    print(f"Accuracy: {np.mean(accuracy)}")

    # plot histogram
    # import matplotlib.pyplot as plt
    # plt.hist(accuracy, bins=20)
    # plt.show()