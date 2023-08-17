import os
import numpy as np
from tqdm import trange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_validate

GROUPING_LEVEL = 1
USE_CROSS_VAL = False
STRATIFIED = True
EXCLUDED_COLUMNS = ["filename"]

suffixes = ["", "_grouped", "_grouped_further"]

filepath = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "split_data", "train",
    "without_hybrids", f"new_results_no_hybrids{suffixes[GROUPING_LEVEL]}.csv")

def main():
    with open(filepath, 'r', newline='', encoding='UTF8') as csvFile:
        headers = csvFile.readline().split(',')

    columns = [
        i for i, header in enumerate(headers) if header not in EXCLUDED_COLUMNS
    ]

    assert len(columns) == len(headers) - len(EXCLUDED_COLUMNS)

    data = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=columns)

    x = data[:, :-1]
    y = data[:, -1]

    if USE_CROSS_VAL:

        kfold = StratifiedKFold(n_splits=5, shuffle=True) if STRATIFIED else KFold(
            n_splits=5, shuffle=True)

        model = LogisticRegression(max_iter=10000)

        scoring = ['accuracy', 'precision', 'recall', 'f1']

        scores = cross_validate(model, x, y, cv=kfold, scoring=scoring)

        print(f'Accuracy:{scores["test_accuracy"].mean(): .3f}')
        print(f'Precision:{scores["test_precision"].mean(): .3f}')
        print(f'Recall:{scores["test_recall"].mean(): .3f}')
        print(f'F1:{scores["test_f1"].mean(): .3f}')

    else:

        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            stratify=y)

        # define and fit model
        model = LogisticRegression(max_iter=10000)
        model.fit(x_train, y_train)

        # evaluate model
        # test_preds = model.predict(x_test)

        # scores = classification_report(y_test,
        #                             test_preds,
        #                             labels=[0, 1],
        #                             target_names=["arm", "zea"],
        #                             digits=3)

        # matrix = confusion_matrix(y_test, test_preds)

        # print(matrix)
        # print(scores)

        accuracy = model.score(x_test, y_test)

        return accuracy

if __name__ == "__main__":
    accuracies = []

    for _ in trange(1000):
        accuracies.append(main())

    print(f"Average Accuracy: {np.mean(accuracies)}")
    print(f"Standard Deviation: {np.std(accuracies)}")