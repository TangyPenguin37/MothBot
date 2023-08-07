### using LDA with a balanced dataset to determine the most important features ###

import os
import numpy as np
import imblearn
from tqdm import tqdm, trange
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

GROUPING_LEVEL = 1
# COLUMNS_TO_USE = [
#     f"{i}_{j}"
#     for i in ["area", "major", "minor", "circ", "feret", "minferet", "ar"] +
#     [f"efd_{k}" for k in range(25)] +
#     [f"colour_{k}_{l}" for k in range(2)
#      for l in ["r", "g", "b"]] + [f"percentage_{k}" for k in range(2)]
#     for j in ["d", "v"]
# ] + ["species\r\n"]

suffixes = ["", "_grouped", "_grouped_further"]

filepath = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "split_data", "train",
    "without_hybrids", f"new_results_no_hybrids{suffixes[GROUPING_LEVEL]}.csv")

with open(filepath, 'r', newline='', encoding='UTF8') as csvFile:
    headers = csvFile.readline().split(',')

def load_data(arm_balance=None,
              balance_train_only=False,
              test_size=0.2,
              columns_to_use=[],
              wing=(0, 0)):

    columns_to_use += ["species\r\n", "location", "side"]
    columns = [
        i for i, header in enumerate(headers) if header in columns_to_use
    ]

    assert headers.index("species\r\n") in columns
    assert len(columns) == len(columns_to_use)

    csvData = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=columns)

    locationID = columns.index(headers.index("location"))
    sideID = columns.index(headers.index("side"))

    # filter by location and side
    csvData = csvData[np.where(csvData[:, locationID] == wing[0])]
    csvData = csvData[np.where(csvData[:, sideID] == wing[1])]

    assert csvData.shape[1] == len(columns_to_use)
    assert csvData.shape[0] > 0

    # add random column
    # csvData = np.stack((np.random.rand(csvData.shape[0]), csvData), axis=1)

    x, y = np.split(csvData, [-1], 1)  # pylint: disable=unbalanced-tuple-unpacking
    y = y.ravel()

    if arm_balance is None:
        return train_test_split(x, y, test_size=test_size, stratify=y)

    assert isinstance(arm_balance, float) and 0 <= arm_balance <= 1

    if balance_train_only:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, stratify=y)
        x_train, y_train = imblearn.over_sampling.SMOTE(  # type: ignore
        ).fit_resample(x_train, y_train)
        return x_train, x_test, y_train, y_test

    x, y = imblearn.over_sampling.SMOTE().fit_resample(x, y)  # type: ignore

    return train_test_split(x, y, test_size=test_size, stratify=y)

def run(train_x, test_x, train_y, test_y, print_report=False):

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
    # run(*load_data(arm_balance=0.5, balance_train_only=False),
    #     print_report=True)

    allColumns = [
        f"{i}_{j}" for i in
        ["area", "major", "minor", "circ", "feret", "minferet", "ar"] +
        [f"efd_{k}" for k in range(25)] +
        [f"colour_{k}_{l}" for k in range(2)
         for l in ["r", "g", "b"]] + [f"percentage_{k}" for k in range(2)]
        for j in ["d", "v"]
    ]

    meanAccuracies = {}

    for wings in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        for col in tqdm(allColumns):
            accuracies = []
            for _ in trange(500, leave=False):
                accuracies.append(
                    run(*load_data(arm_balance=0.5,
                                   balance_train_only=False,
                                   columns_to_use=[col],
                                   wing=wings)))  # type: ignore

            # print(f"\n{col}: {np.mean(accuracies)}")
            meanAccuracies[col] = np.mean(accuracies)

        print("\n\n\n")
        print(f"Wing: {wings}")

        for key, value in sorted(meanAccuracies.items(),
                                 key=lambda item: item[1],
                                 reverse=True):
            print(f"{key}: {value}")