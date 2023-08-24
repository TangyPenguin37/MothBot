import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report
from tqdm import trange

GROUPING_LEVEL = 1
USE_CROSS_VAL = False

suffixes = ["", "_grouped", "_grouped_further"]

filepath = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "split_data", "train",
    "without_hybrids", f"new_results_no_hybrids{suffixes[GROUPING_LEVEL]}.csv")

with open(filepath, 'r', newline='', encoding='UTF8') as csvFile:
    columns = len(csvFile.readline().split(','))

data = np.loadtxt(filepath,
                  delimiter=',',
                  skiprows=1,
                  usecols=range(1, columns))

x = data[:, :-1]
y = data[:, -1]

def run():
    train_x, test_x, train_y, test_y = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y)

    model = RandomForestClassifier()

    if USE_CROSS_VAL:

        kfold = StratifiedKFold(n_splits=5, shuffle=True)

        accuracy = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')
        precision = cross_val_score(model, x, y, cv=kfold, scoring='precision')
        recall = cross_val_score(model, x, y, cv=kfold, scoring='recall')
        f1 = cross_val_score(model, x, y, cv=kfold, scoring='f1')

        print(f'Accuracy:{accuracy.mean(): .3f}')
        print(f'Precision:{precision.mean(): .3f}')
        print(f'Recall:{recall.mean(): .3f}')
        print(f'F1:{f1.mean(): .3f}')

    else:

        model.fit(train_x, train_y)

        # scores = classification_report(test_y,
        #                             model.predict(test_x),
        #                             labels=[0, 1],
        #                             target_names=["arm", "zea"],
        #                             digits=3)

        # print(scores)

        return model.score(test_x, test_y)

if __name__ == "__main__":

    accuracy = [run() for _ in trange(100)]

    print(f"Average accuracy: {np.mean(accuracy)}")
    print(f"Standard deviation: {np.std(accuracy)}")