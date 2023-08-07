### Neural network for classifying arm, hybrid, and zea using the images themselves ###

import os
import csv
import pickle
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from keras import layers
from keras.models import Sequential
from sklearn.model_selection import train_test_split

BATCH_SIZE = 16

print(f'Tensorflow version: {tf.__version__}')

if not os.path.exists('x.pkl') and not os.path.exists('y.pkl'):

    imagesDir = r'data\split_data\train\images\all\cutout'

    filepaths = [
        os.path.join(root, file) for root, _, files in os.walk(imagesDir)
        for file in files
    ]

    with open(r'data\all_info.csv', 'r', encoding='UTF8') as csvFile:
        x = []
        y = []

        IDs = {row[0]: row[1] for row in csv.reader(csvFile)}

        for root, dirs, files in os.walk(imagesDir):
            for file in tqdm(files):
                y.append(IDs[file[:-6]])
                filepath = os.path.join(root, file)
                x.append(
                    tf.image.resize_with_pad(
                        tf.io.decode_jpeg(tf.io.read_file(filepath), channels=3),
                        256, 256))

        mapping = {'arm': 0, 'hyb': 1, 'zea': 2}
        y = list(map(mapping.get, y))

        assert None not in y

    # print(x)
    # print(y)

    pickle.dump(x, open('x.pkl', 'wb'))
    pickle.dump(y, open('y.pkl', 'wb'))

else:

    x = pickle.load(open('x.pkl', 'rb'))
    y = pickle.load(open('y.pkl', 'rb'))

x_train_val, x_test, y_train_val, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.1,
                                                            stratify=y)

x_train, x_val, y_train, y_val = train_test_split(x_train_val,
                                                  y_train_val,
                                                  test_size=(1 / 9),
                                                  stratify=y_train_val)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                     patience=5,
                                     restore_best_weights=True),
    # tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(256, 256, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, )
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# model.summary()

epochs = 100

# history = model.fit(train_ds, epochs=epochs, callbacks=callbacks)
history = model.fit(train_ds,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=val_ds)

# test on test data
test_loss, test_acc = model.evaluate(test_ds, batch_size=BATCH_SIZE)

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
