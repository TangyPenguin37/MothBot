import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

GROUPING_LEVEL = 1

suffixes = ["", "_grouped", "_grouped_further"]
columns = [45, 84, 322]

filepath = os.path.join(
    os.path.dirname(__file__), "../../..", "data", "split_data", "train",
    "without_hybrids",
    f"new_results_no_hybrids{suffixes[GROUPING_LEVEL]}.csv")

csvData = np.loadtxt(filepath,
                     delimiter=',',
                     skiprows=1,
                     usecols=range(1, columns[GROUPING_LEVEL]))

x, y = np.split(csvData, [-1], 1)  # pylint: disable=unbalanced-tuple-unpacking
y = y.ravel()

# only take second column for x
# x = x[:, 2]
# x = x.reshape(-1, 1)

# split data into train and test sets
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(train_x, train_y)

test_preds = lda.predict(test_x)

scores = classification_report(test_y,
                               test_preds,
                               target_names=["arm", "zea"],
                               digits=3)

print(scores)