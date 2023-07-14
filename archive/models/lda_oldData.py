import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

csvData = np.loadtxt(
    'data/original_labelled/data_normalized_onehot_simplified_corrected.csv',
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
print(f'Accuracy: {lda.score(test_x, test_y)}')