""" Just a proof of concept to see if I can figure out what I'm doing! """

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print(f'Tensorflow version: {tf.__version__}')
csvFilepath = r'data\split_data\train\with_hybrids\new_results_grouped.csv'

df = pd.read_csv(csvFilepath)
df.drop(columns=['filename'], inplace=True)

# print(df.head())

# change to onehot
df = pd.get_dummies(df, columns=['species'])

# change label names to arm, hyb, zea
df = df.rename(columns={
    'species_0': 'arm',
    'species_1': 'hyb',
    'species_2': 'zea'
})

df = df.astype('float64')

# normalize data
# TODO: change to only normalize the data that needs to be normalized
max_val = df.max(axis=0)
min_val = df.min(axis=0)

range = max_val - min_val
df = (df - min_val) / range

# print(df)

train_df = df.sample(frac=0.8)
test_df = df.drop(train_df.index)

X_train = train_df.drop(['arm', 'hyb', 'zea'], axis=1)
X_test = test_df.drop(['arm', 'hyb', 'zea'], axis=1)
y_train = train_df[['arm', 'hyb', 'zea']]
y_test = test_df[['arm', 'hyb', 'zea']]

print(X_train)
print(y_train)

input_shape = [X_train.shape[1]]

print(input_shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu',
                          input_shape=input_shape),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='softmax')
])

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                     patience=25,
                                     restore_best_weights=True)

losses = model.fit(X_train, y_train, epochs=1000, validation_split=0.2, callbacks=[callback])

model.evaluate(X_test, y_test)

plt.plot(losses.history['accuracy'])
plt.plot(losses.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()