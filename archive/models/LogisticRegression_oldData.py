import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score

USE_CROSS_VAL = True
USE_GROUPED_DATA = False

# load and prepare data
if USE_GROUPED_DATA:
    columns = list(range(1, 194))
    data = np.loadtxt('data/data_normalized_grouped.csv',
                      delimiter=',',
                      skiprows=1,
                      usecols=columns)
else:
    columns = list(range(1, 51))
    columns.remove(6)
    columns.remove(8)

    data = np.loadtxt('data/data_normalized.csv',
                      delimiter=',',
                      skiprows=1,
                      usecols=columns)

x = data[:, :-1]
y = data[:, -1]

if USE_CROSS_VAL:

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    model = LogisticRegression(max_iter=1000)

    accuracy = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')
    precision = cross_val_score(model, x, y, cv=kfold, scoring='precision')
    recall = cross_val_score(model, x, y, cv=kfold, scoring='recall')
    f1 = cross_val_score(model, x, y, cv=kfold, scoring='f1')

    print(f'Accuracy:{accuracy.mean(): .3f}')
    print(f'Precision:{precision.mean(): .3f}')
    print(f'Recall:{recall.mean(): .3f}')
    print(f'F1:{f1.mean(): .3f}')

else:

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # define and fit model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    # evaluate model
    test_preds = model.predict(x_test)

    scores = classification_report(y_test,
                                   test_preds,
                                   target_names=["arm", "zea"],
                                   digits=3)

    matrix = confusion_matrix(y_test, test_preds)

    print(scores)
    print(matrix)
