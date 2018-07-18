from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import os
from keras.callbacks import ModelCheckpoint
from io import open
import requests
import shutil
from zipfile import ZipFile
import keras
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, GlobalAvgPool2D, BatchNormalization, add, Input
from keras.models import Model
from tensorflow.python.keras.preprocessing import image
import numpy as np
import json

execution_path = os.getcwd()

# ----------------- The Section Responsible for Downloading the Dataset ---------------------


SOURCE_PATH = "https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-jpg.zip"
FILE_DIR = os.path.join(execution_path, "idenprof-jpg.zip")
DATASET_DIR = os.path.join(execution_path, "idenprof")
DATASET_TRAIN_DIR = os.path.join(DATASET_DIR, "train")
DATASET_TEST_DIR = os.path.join(DATASET_DIR, "test")


def download_idenprof():
    if (os.path.exists(FILE_DIR) == False):
        print("Downloading idenprof-jpg.zip")
        data = requests.get(SOURCE_PATH,
                            stream=True)

        with open(FILE_DIR, "wb") as file:
            shutil.copyfileobj(data.raw, file)
        del data

        extract = ZipFile(FILE_DIR)
        extract.extractall(execution_path)
        extract.close()


# ----------------- The Section Responsible for Training ResNet50 on the IdenProf dataset ---------------------

# Directory in which to create models
save_direc = os.path.join(os.getcwd(), 'idenprof_models')

# Name of model files
model_name = 'idenprof_weight_model.{epoch:03d}-{val_acc}.h5'

# Create Directory if it doesn't exist
if not os.path.isdir(save_direc):
    os.makedirs(save_direc)
# Join the directory with the model file
modelpath = os.path.join(save_direc, model_name)

# Checkpoint to save best model
checkpoint = ModelCheckpoint(filepath=modelpath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             period=1)


# Function for adjusting learning rate and saving dummy file
def lr_schedule(epoch):
    """
    Learning Rate Schedule
    """
    # Learning rate is scheduled to be reduced after 80, 120, 160, 180  epochs. Called  automatically  every
    #  epoch as part  of  callbacks  during  training.



    lr = 1e-3
    if epoch > 180:
        lr *= 1e-4
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1

    print('Learning rate: ', lr)
    return lr


lr_scheduler = LearningRateScheduler(lr_schedule)


def resnet_module(input, channel_depth, strided_pool=False):
    residual_input = input
    stride = 1

    if (strided_pool):
        stride = 2
        residual_input = Conv2D(channel_depth, kernel_size=1, strides=stride, padding="same",
                                kernel_initializer="he_normal")(residual_input)
        residual_input = BatchNormalization()(residual_input)

    input = Conv2D(int(channel_depth / 4), kernel_size=1, strides=stride, padding="same",
                   kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(int(channel_depth / 4), kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal")(
        input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(channel_depth, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)

    input = add([input, residual_input])
    input = Activation("relu")(input)

    return input


def resnet_first_block_first_module(input, channel_depth):
    residual_input = input
    stride = 1

    residual_input = Conv2D(channel_depth, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal")(
        residual_input)
    residual_input = BatchNormalization()(residual_input)

    input = Conv2D(int(channel_depth / 4), kernel_size=1, strides=stride, padding="same",
                   kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(int(channel_depth / 4), kernel_size=3, strides=stride, padding="same",
                   kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(channel_depth, kernel_size=1, strides=stride, padding="same", kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)

    input = add([input, residual_input])
    input = Activation("relu")(input)

    return input


def resnet_block(input, channel_depth, num_layers, strided_pool_first=False):
    for i in range(num_layers):
        pool = False
        if (i == 0 and strided_pool_first):
            pool = True
        input = resnet_module(input, channel_depth, strided_pool=pool)

    return input


def ResNet50(input_shape, num_classes=10):
    input_object = Input(shape=input_shape)
    layers = [3, 4, 6, 3]
    channel_depths = [256, 512, 1024, 2048]

    output = Conv2D(64, kernel_size=7, strides=2, padding="same", kernel_initializer="he_normal")(input_object)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(output)
    output = resnet_first_block_first_module(output, channel_depths[0])

    for i in range(4):
        channel_depth = channel_depths[i]
        num_layers = layers[i]

        strided_pool_first = True
        if (i == 0):
            strided_pool_first = False
            num_layers = num_layers - 1
        output = resnet_block(output, channel_depth=channel_depth, num_layers=num_layers,
                              strided_pool_first=strided_pool_first)

    output = GlobalAvgPool2D()(output)
    output = Dense(num_classes)(output)
    output = Activation("softmax")(output)

    model = Model(inputs=input_object, outputs=output)

    return model


def train_network():
    download_idenprof()

    print(os.listdir(os.path.join(execution_path, "idenprof")))

    optimizer = keras.optimizers.Adam(lr=0.01, decay=1e-4)
    batch_size = 32
    num_classes = 10
    epochs = 200

    model = ResNet50((224, 224, 3), num_classes=num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    print("Using real time Data Augmentation")
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(
        rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(DATASET_TRAIN_DIR, target_size=(224, 224),
                                                        batch_size=batch_size, class_mode="categorical")
    test_generator = test_datagen.flow_from_directory(DATASET_TEST_DIR, target_size=(224, 224), batch_size=batch_size,
                                                      class_mode="categorical")

    model.fit_generator(train_generator, steps_per_epoch=int(9000 / batch_size), epochs=epochs,
                        validation_data=test_generator,
                        validation_steps=int(2000 / batch_size), callbacks=[checkpoint, lr_scheduler])


# ----------------- The Section Responsible for Inference ---------------------
CLASS_INDEX = None

MODEL_PATH = os.path.join(execution_path, "idenprof_061-0.7933.h5")
JSON_PATH = os.path.join(execution_path, "idenprof_model_class.json")


def preprocess_input(x):
    x *= (1. / 255)

    return x


def decode_predictions(preds, top=5, model_json=""):
    global CLASS_INDEX

    if CLASS_INDEX is None:
        CLASS_INDEX = json.load(open(model_json))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        for i in top_indices:
            each_result = []
            each_result.append(CLASS_INDEX[str(i)])
            each_result.append(pred[i])
            results.append(each_result)

    return results


def run_inference():
    model = ResNet50(input_shape=(224, 224, 3), num_classes=10)
    model.load_weights(MODEL_PATH)

    picture = os.path.join(execution_path, "Haitian-fireman.jpg")

    image_to_predict = image.load_img(picture, target_size=(
        224, 224))
    image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
    image_to_predict = np.expand_dims(image_to_predict, axis=0)

    image_to_predict = preprocess_input(image_to_predict)

    prediction = model.predict(x=image_to_predict, steps=1)

    predictiondata = decode_predictions(prediction, top=int(5), model_json=JSON_PATH)

    for result in predictiondata:
        print(str(result[0]), " : ", str(result[1] * 100))


# run_inference()
train_network()

