""" Neural network for classifying arm, hybrid, and zea using the extracted features from the images """

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import imblearn

from keras import layers
from keras.models import Sequential
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BATCH_SIZE = 16
BALANCER = imblearn.over_sampling.SMOTE()  # type: ignore

print(f'Tensorflow version: {tf.__version__}')

csvFilepath = r'data\split_data\train\with_hybrids\new_results_grouped.csv'
admixtureCSV = r'data\all_info.csv'

def load_data():

    df = pd.read_csv(csvFilepath)

    # change filename to ID
    df['ID'] = df['filename'].apply(lambda x: int(x.replace('CAM', '')))
    df = df.drop(columns=['filename'], inplace=False).astype('float64')

    # normalized_cols = ["location", "side", "circ_d", "circ_v", "species"] + [
    #     f"colour_{i}_{j}_{k}" for i in range(2) for j in ["r", "g", "b"]
    #     for k in ["d", "v"]
    # ] + [f"percentage_{i}_{j}" for i in range(2) for j in ["d", "v"]]

    # # normalise columns not in normalized_cols
    # df[df.columns.difference(normalized_cols)] = df[df.columns.difference(
    #     normalized_cols)].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # split data by ID

    IDs = df[['ID', 'species']].drop_duplicates()

    train_val_IDs, test_IDs = train_test_split(IDs,
                                               test_size=0.1,
                                               stratify=IDs[['species']])

    train_IDs, val_IDs = train_test_split(train_val_IDs,
                                          test_size=(1 / 9),
                                          stratify=train_val_IDs[['species']])

    train_df = df[df['ID'].isin(train_IDs['ID'])].drop(['ID'], axis=1)
    val_df = df[df['ID'].isin(val_IDs['ID'])].drop(['ID'], axis=1)
    test_df = df[df['ID'].isin(test_IDs['ID'])].drop(['ID'], axis=1)

    unnormalized_cols = list(
        filter(
            lambda x: (train_df[x].max() > 1 or train_df[x].min() < 0) and x !=
            'species', train_df.columns))

    train_df_min = train_df[unnormalized_cols].min()
    train_df_max = train_df[unnormalized_cols].max()

    train_df[unnormalized_cols] = train_df[unnormalized_cols].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))

    for col in unnormalized_cols:
        val_df[col] = (val_df[col] - train_df_min[col]) / (train_df_max[col] -
                                                           train_df_min[col])
        test_df[col] = (test_df[col] - train_df_min[col]) / (
            train_df_max[col] - train_df_min[col])

    # train_x, train_y = imblearn.over_sampling.SMOTE(  # type: ignore
    # ).fit_resample(train_df.drop(['species'], axis=1), train_df[['species']])
    # val_x, val_y = imblearn.over_sampling.SMOTE().fit_resample(  # type: ignore
    #     val_df.drop(['species'], axis=1), val_df[['species']])
    # test_x, test_y = imblearn.over_sampling.SMOTE(  # type: ignore
    # ).fit_resample(test_df.drop(['species'], axis=1), test_df[['species']])

    # train_x, train_y = imblearn.under_sampling.RandomUnderSampler(  # type: ignore
    # ).fit_resample(train_df.drop(['species'], axis=1), train_df[['species']])
    # val_x, val_y = imblearn.under_sampling.RandomUnderSampler().fit_resample(  # type: ignore
    #     val_df.drop(['species'], axis=1), val_df[['species']])
    # test_x, test_y = imblearn.under_sampling.RandomUnderSampler(  # type: ignore
    # ).fit_resample(test_df.drop(['species'], axis=1), test_df[['species']])

    train_x, train_y = BALANCER.fit_resample(  # type: ignore
        train_df.drop(['species'], axis=1), train_df[['species']])
    val_x, val_y = BALANCER.fit_resample(  # type: ignore
        val_df.drop(['species'], axis=1), val_df[['species']])
    test_x, test_y = BALANCER.fit_resample(  # type: ignore
        test_df.drop(['species'], axis=1), test_df[['species']])

    # change to onehot
    train_y = pd.get_dummies(train_y, columns=['species'])
    val_y = pd.get_dummies(val_y, columns=['species'])
    test_y = pd.get_dummies(test_y, columns=['species'])

    # change label names to arm, hyb, zea

    renamed_cols = {'species_0': 'arm', 'species_1': 'hyb', 'species_2': 'zea'}

    train_y = train_y.rename(columns=renamed_cols)
    val_y = val_y.rename(columns=renamed_cols)
    test_y = test_y.rename(columns=renamed_cols)

    return (train_x, val_x, test_x, train_y, val_y, test_y)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

epochs = 2000

def train_test_model(X_train,
                     X_val,
                     X_test,
                     y_train,
                     y_val,
                     y_test,
                     verbose=True):

    callbacks = [
        # tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=100,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                             factor=0.33,
                                             patience=25,
                                             verbose=1 if verbose else 0,
                                             mode='auto',
                                             min_delta=0.0001,
                                             cooldown=0,
                                             min_lr=0)
    ]

    # model = Sequential([
    #     layers.Dense(10,
    #                  activation='sigmoid',
    #                  input_shape=(X_train.shape[1], )),
    #     layers.Dense(3)
    # ])

    # model.compile(
    #     optimizer=tf.keras.optimizers.RMSprop(),
    #     # optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.005),
    #     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #     metrics=['accuracy'])

    assert not np.any(np.isnan(X_train))
    assert not np.any(np.isnan(y_train))

    # history = model.fit(X_train,
    #                     y_train,
    #                     epochs=epochs,
    #                     callbacks=callbacks,
    #                     validation_data=(X_val, y_val),
    #                     batch_size=None,
    #                     validation_batch_size=BATCH_SIZE,
    #                     verbose=2 if verbose else 0)  # type: ignore

    # load model from file
    model = tf.keras.models.load_model(
        Path(__file__).resolve().parent / 'savedModels' / Path(__file__).stem /
        '20230810-121503_0.738 (BEST).h5')

    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]  # type: ignore

    y_pred = model.predict(
        X_test,  # type: ignore
        batch_size=BATCH_SIZE,
        verbose=2 if verbose else 0)  # type: ignore

    # print confusion matrix and classification report
    y_pred = tf.argmax(y_pred, axis=1)
    y_test = tf.argmax(y_test, axis=1)

    if verbose:
        # print(confusion_matrix(y_test, y_pred))
        print(
            classification_report(y_test,
                                  y_pred,
                                  target_names=['arm', 'hyb', 'zea']))

    # print(model.weights)

    # return (history, test_loss, test_acc)

    # save model

    # model.save(
    #     Path(__file__).resolve().parent / 'savedModels' / Path(__file__).stem /
    #     f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{test_acc:.3f}.h5"
    # )

    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # if verbose:
    #     epochs_range = range(len(acc))
    #     plt.figure(figsize=(8, 8))
    #     plt.subplot(1, 2, 1)
    #     plt.plot(epochs_range, acc, label='Training Accuracy')
    #     plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    #     plt.legend(loc='lower right')
    #     plt.title('Training and Validation Accuracy')

    #     plt.subplot(1, 2, 2)
    #     plt.plot(epochs_range, loss, label='Training Loss')
    #     plt.plot(epochs_range, val_loss, label='Validation Loss')
    #     plt.legend(loc='upper right')
    #     plt.title('Training and Validation Loss')
    #     plt.show()

    # return test_acc, y_pred

    return confusion_matrix(y_test, y_pred), classification_report(
        y_test, y_pred, target_names=['arm', 'hyb', 'zea'], output_dict=True)

def main():
    accuracy = []

    for _ in trange(10):
        test_acc, _ = train_test_model(*load_data(), verbose=False)
        accuracy.append(test_acc)

    print(f'Average accuracy: {np.mean(accuracy)}')
    print(f'Standard deviation: {np.std(accuracy)}')

def scores_run():
    final_confusion_matrix = np.zeros((3, 3))
    final_classification_report = {
        "arm": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
        },
        "hyb": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
        },
        "zea": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
        }
    }

    for _ in trange(100):
        result = train_test_model(*load_data(), verbose=False)
        final_confusion_matrix += result[0]
        for key, value in final_classification_report.items():
            for metric in value:
                final_classification_report[key][metric] += float(result[1][key][metric])

    # final_confusion_matrix /= 100
    for key, value in final_classification_report.items():
        for metric in value:
            final_classification_report[key][metric] /= 100

    print(final_confusion_matrix)
    print(final_classification_report)

if __name__ == '__main__':
    # train_test_model(*load_data())
    # main()
    scores_run()