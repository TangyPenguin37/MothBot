### Neural network for classifying arm, hybrid, and zea using the extracted features from the images ###
### DOESN'T REALLY WORK ###

import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import imblearn

from matplotlib import pyplot as plt
from keras import layers
from keras.models import Sequential
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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

    df = df.dropna()

    # normalize data

    unnormalized_cols = list(
        filter(lambda x: df[x].max() > 1 or df[x].min() < 0, df.columns))

    # df[unnormalized_cols] = df[unnormalized_cols].apply(
    #     lambda x: (x - x.min()) / (x.max() - x.min()))

    df[unnormalized_cols] = df[unnormalized_cols].apply(
        lambda x: (x - x.mean()) / x.std())

    # add column to df as bucketized admixture: between 0 and 0.1, 0,1 and 0.9, 0.9 and 1
    df['species'] = pd.cut(df['admixture'],
                           bins=[0, 0.1, 0.9, 1],
                           labels=[0, 1, 2])

    # split data by ID

    IDs = df[['ID', 'species']].drop_duplicates()

    train_val_IDs, test_IDs = train_test_split(IDs,
                                               test_size=0.1,
                                               stratify=IDs['species'])

    train_IDs, val_IDs = train_test_split(train_val_IDs,
                                          test_size=(1 / 9),
                                          stratify=train_val_IDs['species'])

    train_df = df[df['ID'].isin(train_IDs['ID'])].drop(['ID'], axis=1)
    val_df = df[df['ID'].isin(val_IDs['ID'])].drop(['ID'], axis=1)
    test_df = df[df['ID'].isin(test_IDs['ID'])].drop(['ID'], axis=1)

    # balance training data using SMOTENC so that there are equal numbers of each species
    sm = imblearn.over_sampling.SMOTE(sampling_strategy='auto',
                                      random_state=42,
                                      k_neighbors=5,
                                      n_jobs=4)

    X_train = train_df.drop(columns=['species'])
    y_train = train_df[['species']]
    X_train, y_train = sm.fit_resample(X_train, y_train)  # type: ignore

    train_df = pd.concat([X_train, y_train], axis=1)

    X_val = val_df.drop(columns=['species'])
    y_val = val_df[['species']]
    X_val, y_val = sm.fit_resample(X_val, y_val)  # type: ignore
    val_df = pd.concat([X_val, y_val], axis=1)

    return (train_df.drop(columns=['species', 'admixture']),
            val_df.drop(columns=['species', 'admixture']),
            test_df.drop(columns=['species', 'admixture']),
            train_df[['species', 'admixture']],
            val_df[['species', 'admixture']], test_df[['species',
                                                       'admixture']])

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                     patience=100,
                                     restore_best_weights=True),
    # tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

epochs = 2000

def train_test_model(X_train, X_val, X_test, y_train, y_val, y_test):
    model = Sequential([
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    # continuous model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mse'])

    assert not np.any(np.isnan(X_train))
    # assert not np.any(np.isnan(y_train))

    # print(X_train.shape)
    # print(y_train.shape)

    # raise Exception

    history = model.fit(X_train,
                        y_train['admixture'],
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val['admixture']),
                        batch_size=None,
                        validation_batch_size=BATCH_SIZE,
                        verbose=2)  # type: ignore

    # test_acc = model.evaluate(X_test, y_test, verbose=0)[1]  # type: ignore

    # y_pred = tf.argmax(y_pred, axis=1)
    # y_test = tf.argmax(y_test, axis=1)

    y_preds = model.predict(X_test, batch_size=None, verbose=0)  # type: ignore
    y_preds_admixture = y_preds

    # bucketize y_preds
    y_preds = pd.cut(y_preds.flatten(),
                     bins=[-np.inf, 0.1, 0.9, np.inf],
                     labels=[0, 1, 2])

    y_preds = y_preds.codes

    assert not np.any(np.isnan(y_preds))

    y_test_admixture = y_test['admixture']
    y_test = y_test['species']

    print(confusion_matrix(y_test, y_preds))
    print(classification_report(y_test, y_preds))

    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(loss))

    plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')

    # plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # plot true vs predicted admixture
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_admixture, y_preds_admixture)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])

    plt.show()

    return accuracy_score(y_test, y_preds)

def main():
    accuracy = []

    for _ in trange(10):
        test_acc = train_test_model(*load_data())
        accuracy.append(test_acc)

    print(f'Average accuracy: {sum(accuracy) / len(accuracy)}')

if __name__ == '__main__':
    train_test_model(*load_data())
    # main()