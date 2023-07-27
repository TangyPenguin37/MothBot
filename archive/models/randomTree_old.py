import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# load and prepare data
columns = list(range(1, 54))

data = np.loadtxt('data/labelled/data_normalized_onehot.csv',
                    delimiter=',',
                    skiprows=1,
                    usecols=columns)

x = data[:, :-1]
y = data[:, -1]

kfold = StratifiedKFold(n_splits=5, shuffle=True)
model = RandomForestClassifier()

accuracy = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')
precision = cross_val_score(model, x, y, cv=kfold, scoring='precision')
recall = cross_val_score(model, x, y, cv=kfold, scoring='recall')
f1 = cross_val_score(model, x, y, cv=kfold, scoring='f1')

print(f'Accuracy:{accuracy.mean(): .3f}')
print(f'Precision:{precision.mean(): .3f}')
print(f'Recall:{recall.mean(): .3f}')
print(f'F1:{f1.mean(): .3f}')