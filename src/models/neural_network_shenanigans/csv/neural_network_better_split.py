### Neural network for classifying arm, hybrid, and zea using the extracted features from the images ###

import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import layers
from keras.models import Sequential
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BATCH_SIZE = 16

print(f'Tensorflow version: {tf.__version__}')

csvFilepath = r'data\split_data\train\with_hybrids\new_results_grouped.csv'
admixtureCSV = r'data\all_info.csv'

def load_data():

    df = pd.read_csv(csvFilepath)

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

    # change filename to ID
    df['ID'] = df['filename'].apply(lambda x: int(x.replace('CAM', '')))
    df = df.drop(columns=['filename'], inplace=False).astype('float64')

    normalized_cols = ["location", "side", "circ_d", "circ_v", "species"] + [
        f"colour_{i}_{j}_{k}" for i in range(2) for j in ["r", "g", "b"]
        for k in ["d", "v"]
    ] + [f"percentage_{i}_{j}" for i in range(2) for j in ["d", "v"]]

    # normalise columns not in normalized_cols
    df[df.columns.difference(normalized_cols)] = df[df.columns.difference(
        normalized_cols)].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # split data by ID

    IDs = df[['ID', 'arm', 'hyb', 'zea']].drop_duplicates()

    train_val_IDs, test_IDs = train_test_split(
        IDs, test_size=0.1, stratify=IDs[['arm', 'hyb', 'zea']])

    train_IDs, val_IDs = train_test_split(
        train_val_IDs,
        test_size=(1 / 9),
        stratify=train_val_IDs[['arm', 'hyb', 'zea']])

    train_df = df[df['ID'].isin(train_IDs['ID'])].drop(['ID'], axis=1)
    val_df = df[df['ID'].isin(val_IDs['ID'])].drop(['ID'], axis=1)
    test_df = df[df['ID'].isin(test_IDs['ID'])].drop(['ID'], axis=1)

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
                                     patience=10,
                                     restore_best_weights=True),
    # tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

epochs = 1000

def train_test_model(X_train, X_val, X_test, y_train, y_val, y_test):
    model = Sequential([
        layers.Dense(2048,
                     activation='relu',
                     input_shape=[len(X_train.keys())]),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(3)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    assert not np.any(np.isnan(X_train))
    assert not np.any(np.isnan(y_train))

    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val),
                        batch_size=None,
                        validation_batch_size=BATCH_SIZE)

    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]  # type: ignore

    y_pred = model.predict(X_test, batch_size=BATCH_SIZE)

    # print confusion matrix and classification report
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

    for _ in trange(10):
        test_acc = train_test_model(*load_data())
        accuracy.append(test_acc)

    print(f'Average accuracy: {sum(accuracy) / len(accuracy)}')

if __name__ == '__main__':
    train_test_model(*load_data())
    # main()