""" Neural network for classifying arm, hybrid, and zea using the images themselves """

import os
import csv
import numpy as np
import tensorflow as tf

from keras import layers
from keras.models import Sequential

USE_GPU = True
BATCH_SIZE = 16

print(f'Tensorflow version: {tf.__version__}')

imagesDir = r'data\split_data\train\images\all\cutout'

with open(r'data\all_info.csv', 'r', encoding='UTF8') as csvFile:
    labels = []

    IDs = {row[0]: row[1] for row in csv.reader(csvFile)}

    for root, dirs, files in os.walk(imagesDir):
        for file in files:
            labels.append(IDs[file[:-6]])

    mapping = {'arm': 0, 'hyb': 1, 'zea': 2}
    labels = list(map(mapping.get, labels))

    assert None not in labels

train_val_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
    imagesDir,
    labels=labels,  # type: ignore
    validation_split=0.1,
    subset="both",
    batch_size=BATCH_SIZE,  # type: ignore
    seed=np.random.randint(1e6))  # type: ignore

AUTOTUNE = tf.data.AUTOTUNE

train_val_ds = train_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

train_val_ds = train_val_ds.shuffle(1000)
test_ds = test_ds.shuffle(1000)

train_ds = train_val_ds.skip(train_val_ds.cardinality() // 9)
val_ds = train_val_ds.take(train_val_ds.cardinality() // 9)

# train_ds = train_val_ds

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

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
