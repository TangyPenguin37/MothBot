""" Attempted splitting data into train and test sets by ID instead of by image, so that different images from the same individual are not in both the train and test sets. """

import os
import numpy as np
from tqdm import trange
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

GROUPING_LEVEL = 1
EXCLUDED_COLUMNS = []
TEST_SIZE = 0.2

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

csvData = np.loadtxt(
    filepath,
    delimiter=',',
    skiprows=1,
    usecols=columns,
    converters={
        0:
        lambda s: s.replace('CAM', '').replace('_d', '').replace('_v', '').
        replace('.JPG', '').replace('.jpg', '')
    },
    encoding=None)

x, y = np.split(csvData, [-1], 1)  # pylint: disable=unbalanced-tuple-unpacking
y = y.ravel()

def split(test_size=TEST_SIZE):

    # train_x, test_x, train_y, test_y = np.array([]), np.array([]), np.array([]), np.array([])
    train_x, test_x, train_y, test_y = [], [], [], []

    for i in range(3):
        IDs = set(csvData[csvData[:, -1] == i][:, 0])
        print(f"Number of {i} samples: {len(IDs)}")

        train, test = train_test_split(list(IDs), test_size=test_size)

        train_x.extend(csvData[np.isin(csvData[:, 0], train)][:, 0:-1])
        train_y.extend(csvData[np.isin(csvData[:, 0], train)][:, -1])

        test_x.extend(csvData[np.isin(csvData[:, 0], test)][:, 0:-1])
        test_y.extend(csvData[np.isin(csvData[:, 0], test)][:, -1])

    train_x = np.array(train_x)
    test_x = np.array(test_x)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    assert len(train_x) == len(train_y)
    assert len(test_x) == len(test_y)

    assert len(train_x) + len(test_x) == len(csvData)
    assert len(train_y) + len(test_y) == len(csvData)

    # assert that no value in column 0 of train_x is in column 0 of test_x
    assert len(set(train_x[:, 0]).intersection(set(test_x[:, 0]))) == 0

    test_x = test_x[:, 1:]
    train_x = train_x[:, 1:]

    return train_x, test_x, train_y, test_y

def run(print_report=False):

    train_x, test_x, train_y, test_y = split()

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(train_x, train_y)

    if print_report:
        print(
            classification_report(test_y,
                                  lda.predict(test_x),
                                  labels=[0, 1, 2],
                                  target_names=["arm", "hyb", "zea"],
                                  digits=3))

    return lda.score(test_x, test_y)

def measure_accuracy():
    accuracy = []
    for _ in trange(1000):
        accuracy.append(run())

    print(f"Accuracy: {np.mean(accuracy)}")

if __name__ == "__main__":

    run(print_report=True)