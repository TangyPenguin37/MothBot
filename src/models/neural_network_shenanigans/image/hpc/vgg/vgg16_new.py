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

BATCH_SIZE = 32
print(f'Tensorflow version: {tf.__version__}')

imagesDir = 'data/split_data/train/images/all/cutout_resized'

def main():
    with open('data/all_info.csv', 'r', encoding='UTF8') as csvFile:
        labels = []

        IDs = {row[0]: row[1] for row in csv.reader(csvFile)}

        for _, _, files in os.walk(imagesDir):
            for file in files:

                labels.append(IDs[file[:-6]])

        mapping = {'arm': 0, 'hyb': 1, 'zea': 2}
        labels = list(map(mapping.get, labels))

        assert None not in labels

    ds = tf.keras.utils.image_dataset_from_directory(
        imagesDir,
        labels=labels,  # type: ignore
        batch_size=None,  # type: ignore
        seed=np.random.randint(1e6))  # type: ignore

    # preprocess images
    ds = ds.map(lambda x, y: (preprocess_input(x), y))  # type: ignore

    print("Dataset created")

    AUTOTUNE = tf.data.AUTOTUNE

    ds = ds.cache().prefetch(buffer_size=AUTOTUNE)  # type: ignore

    x = [x for x, _ in ds]
    y = [y for _, y in ds]

    x = np.array(x)

    # reshape (4484, 256, 256, 3) to (4484, 196608)
    # x = x.reshape(x.shape[0], -1)
    # x = np.concatenate((x, np.array(parameters)), axis=1)

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
    
    def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
        """
        Compiles a model integrated with VGG16 pretrained layers
        
        input_shape: tuple - the shape of input images (width, height, channels)
        n_classes: int - number of classes for the output layer
        optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
        fine_tune: int - The number of pre-trained layers to unfreeze.
                    If set to 0, all pretrained layers will freeze during training
        """

        # Pretrained convolutional layers are loaded using the Imagenet weights.
        # Include_top is set to False, in order to exclude the model's fully-connected layers.
        conv_base = VGG16(include_top=False,
                        weights='imagenet',
                        input_shape=input_shape)

        # Defines how many layers to freeze during training.
        # Layers in the convolutional base are switched from trainable to non-trainable
        # depending on the size of the fine-tuning parameter.
        if fine_tune > 0:
            for layer in conv_base.layers[:-fine_tune]:
                layer.trainable = False
        else:
            for layer in conv_base.layers:
                layer.trainable = False

        # Create a new 'top' of the model (i.e. fully-connected layers).
        # This is 'bootstrapping' a new top_model onto the pretrained layers.
        top_model = conv_base.output
        top_model = Flatten(name="flatten")(top_model)
        top_model = Dense(4096, activation='relu')(top_model)
        top_model = Dense(1072, activation='relu')(top_model)
        top_model = Dropout(0.2)(top_model)
        output_layer = Dense(n_classes, activation='softmax')(top_model)

        # Group the convolutional base and new fully-connected layers into a Model object.
        model = Model(inputs=conv_base.input, outputs=output_layer)

        # Compiles the model for training.
        model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        return model

    input_shape = (256, 256, 3)
    optim_1 = Adam(learning_rate=0.001)
    n_classes = 3

    n_epochs = 1000

    # First we'll train the model without Fine-tuning
    vgg_model = create_model(input_shape, n_classes, optim_1, fine_tune=3)

    # vgg_model.summary()

    # ModelCheckpoint callback - save best weights
    # tl_checkpoint_1 = ModelCheckpoint(filepath='tl_model_v1.weights.best.hdf5',
    #                                 save_best_only=True,
    #                                 verbose=1)

    # EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss',
                            patience=50,
                            restore_best_weights=True,
                            mode='min')

    vgg_model.fit(
        train_ds,
        epochs=n_epochs,
        validation_data=val_ds,
        # steps_per_epoch=n_steps,
        # validation_steps=n_val_steps,
        callbacks=[early_stop],
        verbose=0)  # type: ignore

    # vgg_model.load_weights('tl_model_v1.weights.best.hdf5') # initialize the best trained weights

    # true_classes = test_ds.classes
    true_classes = np.argmax(y_test, axis=1)

    vgg_preds = vgg_model.predict(test_ds, verbose=2)  # type: ignore
    vgg_pred_classes = np.argmax(vgg_preds, axis=1)

    return accuracy_score(true_classes, vgg_pred_classes)

accuracies = []

for _ in trange(100):
    accuracies.append(main())

print(f"Average Accuracy: {np.mean(accuracies)}")
print(f"Standard Deviation: {np.std(accuracies)}")
