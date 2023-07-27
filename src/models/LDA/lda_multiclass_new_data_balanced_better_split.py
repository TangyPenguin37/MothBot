### Attempted both splitting into test and train sets by ID, as well as balancing the data by oversampling the minority classes, although this worsened the results, likely as the inclusion of hybrids lessened the imbalance in the data. ###

import os
import numpy as np
import imblearn
from tqdm import trange
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# BAD!!!

GROUPING_LEVEL = 1
EXCLUDED_COLUMNS = []

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

def split_data(test_size=0.2):

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

    # assert that ID train_x is in of test_x
    assert len(set(train_x[:, 0]).intersection(set(test_x[:, 0]))) == 0

    test_x = test_x[:, 1:]
    train_x = train_x[:, 1:]

    train_x, train_y = imblearn.over_sampling.SMOTE().fit_resample(train_x, train_y) # type: ignore

    return train_x, test_x, train_y, test_y

# def balance(x, y, arm_split=1/3, hyb_split=1/3):

#     assert arm_split + hyb_split < 1

#     y = y.astype(int)

#     arm_count, hyb_count, zea_count = np.bincount(y)
#     min_count = min(arm_count, hyb_count, zea_count)

#     arm_indices = np.random.choice(
#         np.where(y == 0)[0], int(min_count * arm_split * 3), replace=False)
#     hyb_indices = np.random.choice(
#         np.where(y == 1)[0], int(min_count * hyb_split * 3), replace=False)
#     zea_indices = np.random.choice(
#         np.where(y == 2)[0], int(min_count * (1 - arm_split - hyb_split) * 3), replace=False)

#     indices = np.concatenate((arm_indices, hyb_indices, zea_indices))

#     return x[indices], y[indices]

def run(print_report=False):

    train_x, test_x, train_y, test_y = split_data()

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(train_x, train_y) # type: ignore

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