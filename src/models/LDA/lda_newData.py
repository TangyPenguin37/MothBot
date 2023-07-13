import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

csvData = np.loadtxt('data/data_formatted_no_hybrids_normalized.csv',
                     delimiter=',',
                     skiprows=1,
                     usecols=range(1, 52))

x, y = np.split(csvData, [-1], 1)  # pylint: disable=unbalanced-tuple-unpacking
y = y.ravel()

# only take second column for x
# x = x[:, 2]
# x = x.reshape(-1, 1)

# split data into train and test sets
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(train_x, train_y)

# test
print(f'Accuracy:{lda.score(test_x, test_y): .3f}')

# print f1 score
from sklearn.metrics import f1_score

test_preds = lda.predict(test_x)
print(f'F1:{f1_score(test_y, test_preds): .3f}')

# print confusion matrix

from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(test_y, test_preds)
print(matrix)

# print classification report

from sklearn.metrics import classification_report

scores = classification_report(test_y,
                                 test_preds,
                                    target_names=["arm", "zea"],
                                    digits=3)

print(scores)