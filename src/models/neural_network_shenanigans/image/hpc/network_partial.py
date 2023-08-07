### Neural network for classifying arm, hybrid, and zea using the images themselves ###

import os
import csv
import numpy as np
import tensorflow as tf

from keras import layers
from keras.models import Sequential
from sklearn.model_selection import train_test_split

BATCH_SIZE = 16

print(f'Tensorflow version: {tf.__version__}')
print(os.getcwd())

imagesDir = 'data/split_data/train/images/all/cutout_resized'

with open('data/all_info.csv', 'r', encoding='UTF8') as csvFile:
    labels = []
    indicesToDrop = []

    IDs = {row[0]: row[1] for row in csv.reader(csvFile)}

    for root, dirs, files in os.walk(imagesDir):
        for file in files:

            currentParam = []
            labels.append(IDs[file[:-6]])

            if not ("front" in root):
                indicesToDrop.append(len(labels) - 1)

    mapping = {'arm': 0, 'hyb': 1, 'zea': 2}
    labels = list(map(mapping.get, labels))

    assert None not in labels

ds = tf.keras.utils.image_dataset_from_directory(
    imagesDir,
    labels=labels,  # type: ignore
    batch_size=None,  # type: ignore
    seed=np.random.randint(1e6))  # type: ignore

AUTOTUNE = tf.data.AUTOTUNE

ds = ds.cache().prefetch(buffer_size=AUTOTUNE)  # type: ignore

x = [x for x, _ in ds]
y = [y for _, y in ds]

# drop indices
x = np.delete(x, indicesToDrop, axis=0)
y = np.delete(y, indicesToDrop, axis=0)

print(x.shape)
print(y.shape)

assert x.shape[0] == y.shape[0]

y = tf.keras.utils.to_categorical(y, num_classes=3)

x_train_val, x_test, y_train_val, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.1,
                                                            stratify=y)

x_train, x_val, y_train, y_val = train_test_split(x_train_val,
                                                  y_train_val,
                                                  test_size=(1 / 9),
                                                  stratify=y_train_val)

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(BATCH_SIZE)

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                     patience=25,
                                     restore_best_weights=True),
    # tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

# model = Sequential([
#     layers.Rescaling(1. / 255, input_shape=(256, 256, 3)),
#     # layers.Conv2D(16, 3, padding='same', activation='relu'),
#     # layers.MaxPooling2D(),
#     # layers.Conv2D(32, 3, padding='same', activation='relu'),
#     # layers.MaxPooling2D(),
#     # layers.Conv2D(64, 3, padding='same', activation='relu'),
#     # layers.MaxPooling2D(),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(3, )
# ])

# ALEXNET

model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(256, 256, 3)),
    layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((3, 3), strides=(2, 2)),
    layers.Conv2D(256, (5, 5), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((3, 3), strides=(2, 2)),
    layers.Conv2D(384, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(384, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((3, 3), strides=(2, 2)),
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, )
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.summary()

epochs = 1000

# history = model.fit(train_ds, epochs=epochs, callbacks=callbacks)
history = model.fit(train_ds,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=val_ds,
                    verbose=2)  # type: ignore

# test on test data
test_loss, test_acc = model.evaluate(test_ds, batch_size=BATCH_SIZE,
                                     verbose=2)  # type: ignore

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
