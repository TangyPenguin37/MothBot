""" Neural network for classifying arm, hybrid, and zea using the extracted features from the images """

import datetime
import pandas as pd
import tensorflow as tf
import numpy as np

from keras import layers
from keras.models import Sequential
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BATCH_SIZE = 16

print(f'Tensorflow version: {tf.__version__}')

csvFilepath = r'data\split_data\train\with_hybrids\new_results_grouped.csv'
# csvFilepath = r'data\split_data\train\without_hybrids\new_results_no_hybrids_grouped.csv'

def load_data():

    df = pd.read_csv(csvFilepath)
    df.drop(columns=['filename'], inplace=True)

    # print(df.head())

    # change to onehot
    df = pd.get_dummies(df, columns=['species'])

    # change label names to arm, hyb, zea
    df = df.rename(
        columns={
            'species_0': 'arm',
            'species_1': 'hyb',
            'species_2': 'zea'
            # 'species_1': 'zea'
        })

    df = df.astype('float64')

    normalized_cols = ["location", "side", "circ_d", "circ_v", "species"] + [
        f"colour_{i}_{j}_{k}" for i in range(2) for j in ["r", "g", "b"]
        for k in ["d", "v"]
    ] + [f"percentage_{i}_{j}" for i in range(2) for j in ["d", "v"]]

    # normalise columns not in normalized_cols
    df[df.columns.difference(normalized_cols)] = df[df.columns.difference(
        normalized_cols)].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # max_val = df.max(axis=0)
    # min_val = df.min(axis=0)

    # valRange = max_val - min_val
    # df = (df - min_val) / valRange

    # stratify by arm, hyb, zea
    train_val_df, test_df = train_test_split(df,
                                             test_size=0.1,
                                             stratify=df[['arm', 'hyb',
                                                          'zea']])

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=(1 / 9),
        stratify=train_val_df[['arm', 'hyb', 'zea']])

    return (train_df.drop(['arm', 'hyb', 'zea'],
                          axis=1), val_df.drop(['arm', 'hyb', 'zea'], axis=1),
            test_df.drop(['arm', 'hyb', 'zea'],
                         axis=1), train_df[['arm', 'hyb', 'zea']],
            val_df[['arm', 'hyb', 'zea']], test_df[['arm', 'hyb', 'zea']])

    # X_train = train_df.drop(['arm', 'zea'], axis=1)
    # X_test = test_df.drop(['arm', 'zea'], axis=1)
    # y_train = train_df[['arm', 'zea']]
    # y_test = test_df[['arm', 'zea']]

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                     patience=25,
                                     restore_best_weights=True),
    # tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

epochs = 100

def train_test_model(X_train, X_val, X_test, y_train, y_val, y_test):
    model = Sequential([
        layers.Dense(1024,
                     activation='relu',
                     input_shape=[len(X_train.keys())]),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(3)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # model.summary()
    # history = model.fit(train_ds, epochs=epochs, callbacks=callbacks)
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val),
                        batch_size=BATCH_SIZE,
                        validation_batch_size=BATCH_SIZE)

    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]  # type: ignore

    # print confusion matrix and classification report
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred = tf.argmax(y_pred, axis=1)
    y_test = tf.argmax(y_test, axis=1)

    # print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # print(model.weights)

    # return (history, test_loss, test_acc)

    return test_acc

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# # plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# # plt.show()

def main():
    accuracy = []

    for _ in trange(100):
        test_acc = train_test_model(*load_data())
        accuracy.append(test_acc)

    print(f"Average Accuracy: {np.mean(accuracy)}")
    print(f"Standard Deviation: {np.std(accuracy)}")

if __name__ == '__main__':
    # train_test_model(*load_data())
    main()