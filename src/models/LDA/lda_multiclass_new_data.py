""" LDA using the new (slightly different) data from the Python scripts rather than the original data from the ImageJ macros """

import os
import numpy as np
from tqdm import trange
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

GROUPING_LEVEL = 1
EXCLUDED_COLUMNS = ["filename", "location", "side"]

suffixes = ["", "_grouped", "_grouped_further"]

filepath = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data",
                        "split_data", "train", "with_hybrids",
                        f"new_results{suffixes[GROUPING_LEVEL]}.csv")

with open(filepath, 'r', newline='', encoding='UTF8') as csvFile:
    headers = csvFile.readline().split(',')

columns = [
    i for i, header in enumerate(headers) if header not in EXCLUDED_COLUMNS
]

assert len(columns) == len(headers) - len(EXCLUDED_COLUMNS)

csvData = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=columns)

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

    # return confusion_matrix(test_y, lda.predict(test_x))
    return lda.score(test_x, test_y)

if __name__ == "__main__":
    accuracy = []
    for _ in trange(1000):
        accuracy.append(run())

    print(f"Accuracy: {np.mean(accuracy)}")

    # run(print_report=True)

# if __name__ == "__main__":
#     confusion_matrices = np.zeros((3, 3))

#     for _ in trange(1000):
#         confusion_matrices += run()

#     print(confusion_matrices)