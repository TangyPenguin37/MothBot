### using multiple GPUs to speed up training ###

import os
import csv
import tensorflow as tf

from tqdm import trange
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BATCH_SIZE = 16

IMAGE_DIR = 'data/split_data/train/images/all/cutout_resized'

INPUT_SHAPE = (256, 256, 3)
CLASSES = 3
EPOCHS = 1000

MIRRORED_STRATEGY = tf.distribute.MirroredStrategy()

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

with open('data/all_info.csv', 'r', encoding='UTF8') as csvFile:
    labels = []

    IDs = {row[0]: row[1] for row in csv.reader(csvFile)}

    for _, _, files in os.walk(IMAGE_DIR):
        for file in files:

            labels.append(IDs[file[:-6]])

    mapping = {'arm': 0, 'hyb': 1, 'zea': 2}
    labels = list(map(mapping.get, labels))

    assert None not in labels

ds = tf.keras.utils.image_dataset_from_directory(
    IMAGE_DIR,
    labels=labels,  # type: ignore
    batch_size=None,  # type: ignore
    seed=np.random.randint(1e6))  # type: ignore

# preprocess images
ds = ds.map(lambda x, y: (preprocess_input(x), y))  # type: ignore

print("Dataset created")

AUTOTUNE = tf.data.AUTOTUNE

# ds = ds.cache().prefetch(buffer_size=AUTOTUNE)  # type: ignore

x = [x for x, _ in ds]
y = [y for _, y in ds]

x = np.array(x)
y = tf.keras.utils.to_categorical(y, num_classes=3)

with MIRRORED_STRATEGY.scope():

    OPTIMIZER = Adam(learning_rate=0.001)

    conv_base = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=INPUT_SHAPE)

    for layer in conv_base.layers[:-3]:
        layer.trainable = False

    model = Model(inputs=conv_base.input,
                  outputs=Dense(CLASSES, activation='softmax')(Dropout(0.2)(
                      Dense(1072,
                            activation='relu')(Dense(4096, activation='relu')(
                                Flatten(name="flatten")(conv_base.output))))))

    model.compile(optimizer=OPTIMIZER,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

def run():

    x_train_val, x_test, y_train_val, y_test = train_test_split(x,
                                                                y,
                                                                test_size=0.1,
                                                                stratify=y)

    x_train, x_val, y_train, y_val = train_test_split(x_train_val,
                                                      y_train_val,
                                                      test_size=(1 / 9),
                                                      stratify=y_train_val)

    # distribute dataset
    train_ds = MIRRORED_STRATEGY.experimental_distribute_dataset(
        tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).repeat(100).shuffle(10000).batch(BATCH_SIZE))
    val_ds = MIRRORED_STRATEGY.experimental_distribute_dataset(
        tf.data.Dataset.from_tensor_slices(
            (x_val, y_val)).repeat(100).batch(BATCH_SIZE))
    # test_ds = MIRRORED_STRATEGY.experimental_distribute_dataset(
    #     tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE))

    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(BATCH_SIZE)

    model.fit(train_ds,
              epochs=EPOCHS,
              validation_data=val_ds,
              callbacks=[
                  EarlyStopping(monitor='val_loss',
                                patience=50,
                                restore_best_weights=True,
                                mode='min')
              ],
              steps_per_epoch=100,
              validation_steps=100,
              verbose=0)  # type: ignore

    true_classes = np.argmax(y_test, axis=1)

    vgg_preds = model.predict(test_ds, verbose=0)  # type: ignore
    vgg_pred_classes = np.argmax(vgg_preds, axis=1)

    return accuracy_score(true_classes, vgg_pred_classes)

accuracies = []

for _ in trange(100):
    print(f"Iteration {_ + 1}")
    accuracy = run()
    print(f"Accuracy: {accuracy}")
    accuracies.append(accuracy)

print(f"Average Accuracy: {np.mean(accuracies)}")
print(f"Standard Deviation: {np.std(accuracies)}")
