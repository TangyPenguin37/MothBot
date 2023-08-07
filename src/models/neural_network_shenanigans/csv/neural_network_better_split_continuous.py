### Neural network for classifying arm, hybrid, and zea using the extracted features from the images ###

import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import pyplot as plt
from keras import layers
from keras.models import Sequential
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

BATCH_SIZE = 4

print(f'Tensorflow version: {tf.__version__}')

csvFilepath = r'data\split_data\train\with_hybrids\new_results_grouped.csv'
admixtureCSV = r'data\all_info.csv'

def load_data():

    df = pd.read_csv(csvFilepath).drop(columns=['species'], inplace=False)

    # change filename to ID
    df['ID'] = df['filename'].apply(lambda x: int(x.replace('CAM', '')))
    df = df.drop(columns=['filename'], inplace=False).astype('float64')

    # add admixture data
    admixture_df = pd.read_csv(admixtureCSV)
    admixture_df = admixture_df[['id', 'admixture']]
    admixture_df['id'] = admixture_df['id'].apply(
        lambda x: int(x.replace('CAM', '')))
    admixture_df = admixture_df.astype('float64')
    admixture_df = admixture_df.rename(columns={'id': 'ID'})
    df = pd.merge(df, admixture_df, on='ID')

    assert set(df['ID']) <= set(admixture_df['ID'])

    # drop rows with NA admixture
    df = df.dropna()

    normalized_cols = ["location", "side", "circ_d", "circ_v", "admixture"] + [
        f"colour_{i}_{j}_{k}" for i in range(2) for j in ["r", "g", "b"]
        for k in ["d", "v"]
    ] + [f"percentage_{i}_{j}" for i in range(2) for j in ["d", "v"]]

    # normalise columns not in normalized_cols
    df[df.columns.difference(normalized_cols)] = df[df.columns.difference(
        normalized_cols)].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # split data by ID

    IDs = df[['ID']].drop_duplicates()

    train_val_IDs, test_IDs = train_test_split(IDs, test_size=0.1)

    train_IDs, val_IDs = train_test_split(train_val_IDs, test_size=(1 / 9))

    train_df = df[df['ID'].isin(train_IDs['ID'])].drop(['ID'], axis=1)
    val_df = df[df['ID'].isin(val_IDs['ID'])].drop(['ID'], axis=1)
    test_df = df[df['ID'].isin(test_IDs['ID'])].drop(['ID'], axis=1)

    return (train_df.drop(['admixture'],
                          axis=1), val_df.drop(['admixture'], axis=1),
            test_df.drop(['admixture'], axis=1), train_df[['admixture']],
            val_df[['admixture']], test_df[['admixture']])

    # X_train = train_df.drop(['arm', 'zea'], axis=1)
    # X_test = test_df.drop(['arm', 'zea'], axis=1)
    # y_train = train_df[['arm', 'zea']]
    # y_test = test_df[['arm', 'zea']]

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                     patience=100,
                                     restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

epochs = 1000

def train_test_model(X_train, X_val, X_test, y_train, y_val, y_test):
    model = Sequential([
        layers.Dense(2048, activation='gelu', input_shape=[X_train.shape[1]]),
        layers.Dropout(0.2),
        layers.Dense(1024, activation='gelu'),
        layers.Dropout(0.2),
        layers.Dense(512, activation='gelu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='gelu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='gelu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='gelu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='gelu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='gelu'),
        layers.Dense(1, kernel_initializer='normal', activation='sigmoid')
    ])

    # continuous model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mse'])

    assert not np.any(np.isnan(X_train))
    assert not np.any(np.isnan(y_train))

    # print(X_train.shape)
    # print(y_train.shape)

    # raise Exception

    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val),
                        batch_size=None,
                        validation_batch_size=BATCH_SIZE)

    # test_acc = model.evaluate(X_test, y_test, verbose=0)[1]  # type: ignore

    # print confusion matrix and classification report
    # y_pred = tf.argmax(y_pred, axis=1)
    # y_test = tf.argmax(y_test, axis=1)

    y_pred = model.predict(X_test, batch_size=None)
    # print(X_test)
    # print(y_pred)

    assert y_test.shape[0] == y_pred.shape[0]
    assert y_test.shape[1] == y_pred.shape[1]

    # raise Exception

    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

    score = r2_score(y_test, y_pred)

    print(score)

    # plot y_pred vs y_test
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.show()

    return score

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