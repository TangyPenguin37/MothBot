""" Attempted under- and over-sampling classes to rectify class imbalance and improve zea precision and recall """

import os
import numpy as np
import imblearn
from tqdm import trange
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

GROUPING_LEVEL = 1
EXCLUDED_COLUMNS = ["filename"]

suffixes = ["", "_grouped", "_grouped_further"]

filepath = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "split_data", "train",
    "without_hybrids", f"new_results_no_hybrids{suffixes[GROUPING_LEVEL]}.csv")

with open(filepath, 'r', newline='', encoding='UTF8') as csvFile:
    headers = csvFile.readline().split(',')

columns = [
    i for i, header in enumerate(headers) if header not in EXCLUDED_COLUMNS
]

assert headers.index("species\r\n") in columns
assert len(columns) == len(headers) - len(EXCLUDED_COLUMNS)

def load_data(arm_balance=None, balance_train_only=False, test_size=0.2):
    csvData = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=columns)

    x, y = np.split(csvData, [-1], 1)  # pylint: disable=unbalanced-tuple-unpacking
    y = y.ravel()

    if arm_balance is None:
        return train_test_split(x, y, test_size=test_size, stratify=y)

    if not isinstance(arm_balance, float):
        raise TypeError("arm_balance must be a float")

    if balance_train_only:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, stratify=y)
        # x_train, y_train = balance(x_train, y_train, arm_balance)
        x_train, y_train = imblearn.over_sampling.SMOTE(  # type: ignore
        ).fit_resample(x_train, y_train)
        return x_train, x_test, y_train, y_test

    # x, y = balance(x, y, arm_balance)
    x, y = imblearn.over_sampling.SMOTE().fit_resample(x, y)  # type: ignore

    return train_test_split(x, y, test_size=test_size, stratify=y)

def balance(x, y, arm_split):

    y = y.astype(int)

    arm_count, zea_count = np.bincount(y)
    min_count = min(arm_count, zea_count)

    arm_indices = np.random.choice(np.where(y == 0)[0],
                                   int(min_count * arm_split * 2),
                                   replace=False)
    zea_indices = np.random.choice(np.where(y == 1)[0],
                                   int(min_count * (1 - arm_split) * 2),
                                   replace=False)

    indices = np.concatenate((arm_indices, zea_indices))

    print(len(indices))

    return x[indices], y[indices]

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
    data = load_data(arm_balance=0.5, balance_train_only=False)
    run(*data, print_report=True)
    # accuracy = []
    # for _ in trange(1000):
    #     x, y = load_data(split_evenly=False)
    #     accuracy.append(run())

    # print(f"Accuracy: {np.mean(accuracy)}")