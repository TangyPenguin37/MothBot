""" Attempted under- and over-sampling classes to rectify class imbalance and improve zea precision and recall """
""" Doesn't work """

import os
import numpy as np
import imblearn
from tqdm import trange
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

GROUPING_LEVEL = 1
EXCLUDED_COLUMNS = ["filename", "species\r\n"]

suffixes = ["", "_grouped", "_grouped_further"]

filepath = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data",
                        "split_data", "train", "with_hybrids",
                        f"new_results{suffixes[GROUPING_LEVEL]}.csv")

admixtureCSV = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                            "data", "all_info.csv")

with open(filepath, 'r', newline='', encoding='UTF8') as csvFile:
    headers = csvFile.readline().split(',')

columns = [
    i for i, header in enumerate(headers) if header not in EXCLUDED_COLUMNS
]

# assert headers.index("species\r\n") in columns
assert len(columns) == len(headers) - len(EXCLUDED_COLUMNS)

def load_data(test_size=0.2):
    x = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=columns)
    IDs = [
        np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=[0], dtype=str)
    ][0]
    yData = dict(
        np.loadtxt(admixtureCSV,
                   delimiter=',',
                   skiprows=1,
                   usecols=[0, 3],
                   dtype=str))

    # print(IDs.shape)
    # print(x.shape)
    # print(len(yData))

    y = []
    xCopy = []
    for i, val in enumerate(IDs):
        if val not in yData:
            raise ValueError(f"ID {val} not found in admixture CSV")
        if yData[val] == "NA":
            continue
        y.append(float(yData[val]))
        xCopy.append(x[i])

    x = np.array(xCopy)
    y = np.array(y, dtype=float)

    # y = y.reshape(-1, 1)

    # print(x.shape)
    print(y)

    return train_test_split(x, y, test_size=test_size)


# def balance(x, y, arm_split):

#     y = y.astype(int)

#     arm_count, zea_count = np.bincount(y)
#     min_count = min(arm_count, zea_count)

#     arm_indices = np.random.choice(np.where(y == 0)[0],
#                                    int(min_count * arm_split * 2),
#                                    replace=False)
#     zea_indices = np.random.choice(np.where(y == 1)[0],
#                                    int(min_count * (1 - arm_split) * 2),
#                                    replace=False)

#     indices = np.concatenate((arm_indices, zea_indices))

#     return x[indices], y[indices]

def run(train_x, test_x, train_y, test_y):

    lda = LinearDiscriminantAnalysis()
    lda.fit(train_x, train_y)

    y_pred = lda.predict(test_x)

    # convert to numeric

    return r2_score(test_y, y_pred)

if __name__ == "__main__":
    data = load_data()
    print(run(*data))
    # accuracy = []
    # for _ in trange(1000):
    #     x, y = load_data(split_evenly=False)
    #     accuracy.append(run())

    # print(f"Accuracy: {np.mean(accuracy)}")