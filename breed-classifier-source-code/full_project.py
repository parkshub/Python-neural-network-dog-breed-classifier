#-------------------------------------------------------#
# Title: Dog Breed Classifier
# Desc: A convolutional neural network to detect a dog's breed
# Author: Andrew Park
#-------------------------------------------------------#



# /------------------------
# Introduction
# /------------------------

'''
For this project, I set out to create a dog breed classifier (133 breeds) with an accuracy at or above 90% using Python. Throughout this project, I explored various models and model structures, beginning with ResNet-18 and ending with a cocktail of concatenated convolutional neural networks (CCCNN). The initial exploratory stages (sections 1-3) were imperative for my personal growth, although, less interesting. So I’ve condensed these three sections to quickly jump to section 4. Please, stick around until my favorite section, “Jessie and Friends.”

Please visit https://neural-network-breed-classifier.netlify.app/ for the full report.
The dataset can be found here: https://github.com/udacity/dog-project
'''

# /------------------------
# Section 1: Managing dataset
# /------------------------

# https://yasoob.me/posts/understanding-and-writing-jpeg-decoder-in-python/

from struct import unpack
import imghdr
import os

corrupted_1 = []
all_img_path = []

path = "dogImages"

for root, subdirectories, files in os.walk(path):
    for file in files:
        img_path = os.path.join(root, file)

        all_img_path.append(img_path)

        what = imghdr.what(img_path)

        jpeg = (what == 'jpeg')
        if jpeg == False:
            corrupted_1.append(img_path)

[os.remove(x) for x in corrupted_1]

marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while (True):
            marker, = unpack(">H", data[0:2])
            print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2 + lenchunk:]
            if len(data) == 0:
                break


corrupted_2 = []

for img_raw in all_img_path:
    img = JPEG(img_raw)

    try:
        img.decode()
    except:
        corrupted_2.append(img_raw)

for file in corrupted_2:
    os.remove(file)

# /------------------------
# Section 2: ResNet-18
# /------------------------


###### Loading in data #####

from tensorflow.keras.preprocessing import image_dataset_from_directory

IMG_SIZE = (200, 200)
train_dir = "dogImages/train"
valid_dir = "dogImages/valid"

train_dataset = image_dataset_from_directory(
    train_dir,
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=32
)

valid_dataset = image_dataset_from_directory(
    valid_dir,
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=32
)

###### Cleaning up labels #####

import re
import glob
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img


def extract_labels(path):
    labels_raw = [x for x in glob.glob(path)]
    labels = [re.search(r'(?<=\.)\w*', x)[0] for x in labels_raw]
    labels = [re.sub(r'_', ' ', x).title().strip() for x in labels]
    return labels


labels_str = extract_labels('dogImages/train*/*')

for images, labels in train_dataset.take(1):
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        plt.imshow(array_to_img(images[i]))
        plt.title(labels_str[labels[i]])
        plt.axis("off")

###### Data augmentation #####

import tensorflow as tf
from tensorflow.keras.layers import RandomFlip, RandomRotation

data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal_and_vertical'),
    RandomRotation(0.2)])

# graphing 9 random augmented image
for image, label in train_dataset.take(1):
    fig = plt.figure()
    plt.suptitle(labels_str[label[0]])
    for i in range(9):
        augmented_image = data_augmentation(image[0])
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(array_to_img(augmented_image))
        plt.axis("off")

###### Creating and training ResNet-18 #####


import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Input, ZeroPadding2D, MaxPooling2D, \
    AveragePooling2D, Flatten, Dense, Dropout

os.chdir('resnet18')


def conv_block(X, filter, s=2, training=True):
    X_shortcut = X

    # layer 1
    X = Conv2D(filters=filter, kernel_size=1, strides=(s, s), padding='valid')(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    # layer 2
    X = Conv2D(filters=filter, kernel_size=3, strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X, training=training)

    # skip connection
    X_shortcut = Conv2D(filters=filter, kernel_size=1, strides=(s, s), padding='valid')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def id_block(X, filter, training=True):
    X_shortcut = X

    # layer 1
    X = Conv2D(filters=filter, kernel_size=3, padding='same')(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    # layer 2
    X = Conv2D(filters=filter, kernel_size=3, padding='same')(X)
    X = BatchNormalization(axis=3)(X, training=training)

    # skip connection
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def resnet18_func(input_shape=(200, 200, 3), classes=133, drop=0.0, augment=True):
    X_input = Input(input_shape)

    X = X_input

    # Data augmentation
    if augment == True:
        X = data_augmentation(X)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X)

    # Stage 1
    X = Conv2D(filters=64, kernel_size=7, strides=(2, 2))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = conv_block(X, filter=64, s=1)
    X = id_block(X, filter=64)

    # Stage 3
    X = conv_block(X, filter=128, s=2)
    X = id_block(X, filter=128)

    # Stage 4
    X = conv_block(X, filter=256, s=2)
    X = id_block(X, filter=256)

    # Stage 5
    X = conv_block(X, filter=512, s=2)
    X = id_block(X, filter=512)

    # Final step
    X = AveragePooling2D(pool_size=(2, 2))(X)

    X = Flatten()(X)

    X = Dense(512, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(drop)(X)

    X = Dense(256, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(drop)(X)

    X = Dense(256, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(drop)(X)

    X_outputs = Dense(classes, activation='softmax')(X)
    model = Model(inputs=X_input, outputs=X_outputs)

    return model


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

resnet18 = resnet18_func(input_shape=(200, 200, 3), classes=133, drop=0.2, augment=True)
resnet18.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = resnet18.fit(train_dataset, epochs=100, validation_data=valid_dataset)

###### ResNet-18 results #####

import pickle
import numpy as np


def graph_history(*args, main_title='Model', axes_titles=[], xmax=100, ymin=0):
    history_list = [*args]
    num = len(history_list)

    fig, axes = plt.subplots(1, num, sharey=True, figsize=(13, 6))  # original
    fig.text(0.51, 0.04, 'accuracy', ha='center')  # original
    fig.text(0.08, 0.5, 'epoch', va='center', rotation='vertical')  # original

    fig.suptitle(main_title, size=15)

    for i, history in enumerate(history_list):

        if num == 1:
            ax = axes
            last_ax = ax
            adjust = 0.6
            axes_titles = ['']
            i = 0

        else:
            ax = axes[i]
            last_ax = axes[-1]
            adjust = 3

        if num == 3:
            adjust = 4

        ax.set_title(axes_titles[i], loc='left', size=10)

        acc = history['accuracy']
        val_acc = history['val_accuracy']

        ax.plot(acc, label='Training Accuracy')
        ax.plot(val_acc, label='Validation Accuracy')

        ax.set_xlim(0, xmax)
        ax.set_ylim(ymin, 1.01)

        x = np.argmax(history['val_accuracy'])
        y = np.max(history['val_accuracy'])

        ax.hlines(y, xmin=0, xmax=x, linestyles='--', color='r', alpha=0.3)
        ax.vlines(x, ymin=0, ymax=y, linestyles='--', color='r', alpha=0.3, label='Validation Max Accuracy')
        ax.annotate(f'{str(round(y, 3))} @ epoch {x}', (x - adjust, 0.02), rotation='vertical', )  # original

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.83, 0.87))


history1 = pickle.load(open('full_history_resnet18_editeddrop', 'rb'))  # dropout 0.2
history2 = pickle.load(open('full_history_resnet18_edited', 'rb'))  # dropout 0.5
graph_history(history1, history2, main_title='ResNet-18', axes_titles=['dropout rate 0.2', 'dropout rate 0.5'],
              xmax=100)

# /------------------------
# Section 3: ResNet-50, More Power!
# /------------------------

os.chdir('..')
os.chdir('resnet50')

def resnet50_func_basic(input_shape=(200, 200, 3), classes=133, tune_at=164, drop=0.3, reg=0.0, tuning=True):
    base = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)

    if tuning == True:
        for layer in base.layers[:tune_at]:
            layer.trainable = False
    else:
        base.trainable = False

    X_inputs = Input(input_shape)

    # data augmentation
    X = data_augmentation(X_inputs)

    # preprocess
    X = preprocess_input(X)

    # base model
    X = base(X, training=False)
    X = Flatten()(X)

    # norm, dense, drop
    X = BatchNormalization()(X)
    X = Dense(512, kernel_regularizer=regularizers.l2(reg), activation='relu')(X)
    X = Dropout(drop)(X)

    X = BatchNormalization()(X)
    X = Dense(256, kernel_regularizer=regularizers.l2(reg), activation='relu')(X)
    X = Dropout(drop)(X)

    X = BatchNormalization()(X)
    X = Dense(256, kernel_regularizer=regularizers.l2(reg), activation='relu')(X)
    X = Dropout(drop)(X)

    X_outputs = Dense(classes, activation='softmax')(X)

    model = Model(X_inputs, X_outputs)

    return model


def deeper_resnet50_func(input_shape=(200, 200, 3), classes=133, tune_at=164, drop=0.3, reg=0.0, tuning=True):
    base = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)

    if tuning == True:
        for layer in base.layers[:tune_at]:
            layer.trainable = False
    else:
        base.trainable = False

    X_inputs = Input(input_shape)

    # data augmentation
    X = data_augmentation(X_inputs)

    # preprocess
    X = preprocess_input(X)

    # base model
    X = base(X, training=False)

    X = Flatten()(X)

    # norm, dense, drop
    X = BatchNormalization()(X)
    X = Dense(512, kernel_regularizer=regularizers.l2(reg), activation='relu')(X)
    X = Dropout(drop)(X)

    X = BatchNormalization()(X)
    X = Dense(512, kernel_regularizer=regularizers.l2(reg), activation='relu')(X)
    X = Dropout(drop)(X)

    X = BatchNormalization()(X)
    X = Dense(512, kernel_regularizer=regularizers.l2(reg), activation='relu')(X)
    X = Dropout(drop)(X)

    X = BatchNormalization()(X)
    X = Dense(512, kernel_regularizer=regularizers.l2(reg), activation='relu')(X)
    X = Dropout(drop)(X)

    X_outputs = Dense(classes, activation='softmax')(X)

    model = Model(X_inputs, X_outputs)

    return model


##### Training and graphing ResNet-50 #####

# 1
resnet50 = resnet50_func_basic(tuning=False)
resnet50.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
resnet50_history = resnet50.fit(train_dataset, epochs=100, validation_data=valid_dataset)

# 2
resnet50 = resnet50_func_basic(tuning=True)
resnet50.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
resnet50_history = resnet50.fit(train_dataset, epochs=100, validation_data=valid_dataset)

# 3
resnet50 = deeper_resnet50_func(tuning=True)
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
resnet50.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
resnet50_history = resnet50.fit(train_dataset, epochs=100, validation_data=valid_dataset)

# graph figure 3
history1 = pickle.load(open('1.full_history_resnet50_notune', 'rb'))
history2 = pickle.load(open('2.full_history_resnet50_tune', 'rb'))
history3 = pickle.load(open('3.full_history_resnet50_deep_tune_lr', 'rb'))

graph_history(history1, history2, history3, main_title='ResNet-50', axes_titles=['basic,\nno tuning', 'basic,\ntuning',
                                                                                 'deep,\ntuning,\ndecreased lr'])

# /------------------------
# Section 4: Bottleneck Features
# /------------------------


##### Extracting labels #####

os.chdir('bottleneck_section')

# only labels
label_path = 'dogImages - combined*/*'
labels_str = extract_labels(label_path)

# tagets
target_path = 'dogImages - combined*/*/*'
targets_str = extract_labels(target_path)

from sklearn.preprocessing import OrdinalEncoder


def encoder(targets, labels):
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(np.asarray(labels).reshape(-1, 1))
    encoded_targets = ordinal_encoder.transform(np.asarray(targets).reshape(-1, 1))
    return encoded_targets


y = encoder(targets_str, labels_str)

##### Extracting resnet50 bottleneck features #####

from keras.preprocessing.image import load_img


def extract_img(path, size=(200, 200)):
    x = [load_img(x, target_size=size) for x in tqdm(glob.glob(path))]
    x = [np.asarray(x).astype('float32') for x in tqdm(x)]
    x = np.stack(x)
    return x


def extract_bottleneck(path, model, preprocesser, filename='filename', size=(200, 200)):
    images = extract_img(path, size=size)
    shape = size + (3,)
    pre = preprocesser(images)
    model = model(include_top=False, weights='imagenet', input_shape=shape)
    output = model.predict(pre)
    pickle.dump(output, open(filename, 'wb'))
    return output


resnet50_features = extract_bottleneck(target_path, ResNet50, preprocess_input_resnet50, filename='resnet50_features')

# splitting
from sklearn.model_selection import train_test_split

resnet50_train, resnet50_test, y_train, y_test = train_test_split(resnet50_features, y, random_state=42, shuffle=True)


##### Setting up end/top part of neural network #####

def model_top(bottleneck_features):
    he_initializer = tf.keras.initializers.HeNormal()

    x_input = Input(shape=bottleneck_features.shape[1:])
    x = GlobalAveragePooling2D()(x_input)
    x_out = Dense(133, activation='softmax', kernel_initializer=he_initializer)(x)

    model = Model(inputs=x_input, outputs=x_out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


model_resnet50 = model_top(resnet_train)

##### Training ResNet-50 with bottleneck features #####

# callback function
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger


def callbacks_funcv2(main_checkpoint_path, min_delta=0.0, weights=False, best=True):
    main_checkpoint_path = '.\\' + main_checkpoint_path

    checkpoint_name = '\epoch_{epoch:02d}-val_acc_{val_accuracy:.2f}.hdf5'
    checkpoint_path = main_checkpoint_path + checkpoint_name
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_weights_only=weights,
        save_best_only=best)

    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=min_delta, patience=10)

    filename = '\log.csv'
    file_path = main_checkpoint_path + filename
    history_logger = CSVLogger(file_path, append=True)

    callbacks = [checkpoint, history_logger, early_stop]

    return callbacks


callbacks = callbacks_funcv2('checkpoints_resnet50')

history = model_resnet50.fit(
    resnet50_train,
    y_train,
    validation_data=(resnet50_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

##### Results #####

import pandas as pd

history = pd.read_csv('checkpoints_resnet50/log.csv').iloc[:, 1:].to_dict('list')
graph_history(history, main_title='ResNet-50, bottleneck', axes_titles=[], xmax=50)

# /------------------------
# Section 5: A Cocktail of Concatenated Convolutional Neural Networks (CCCNN)
# /------------------------


##### Extracting bottleneck features #####

from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.resnet_v2 import preprocess_input as preprocess_input_resnet152

resnetv2_features = pickle.load(open('resnetv2_features', 'rb'))  # run this instead of the line below
resnetv2_features = extract_bottleneck(target_path, ResNet152V2, preprocess_input_resnetv2,
                                       filename='resnetv2_features')
resnetv2_train, resnetv2_test = train_test_split(resnetv2_features, random_state=42, shuffle=True)

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as preprocess_input_xception

xception_features = pickle.load(open('xception_features', 'rb')) # run this instead of the line below
xception_features = extract_bottleneck(target_path, Xception, preprocess_input_xception, filename='xception_features')
xcep_train, xcep_test = train_test_split(xception_features, random_state=42, shuffle=True)

from keras.applications.efficientnet import EfficientNetB7
from keras.applications.efficientnet import preprocess_input as preprocess_input_efficient

efficient_features = pickle.load(open('efficient_features', 'rb')) # run this instead of the line below
efficient_features = extract_bottleneck(path=target_path, model=EfficientNetB7, preprocesser=preprocess_input_efficient,
                                        filename='efficient_features')
eff_train, eff_test = train_test_split(efficient_features, random_state=42, shuffle=True)

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_incep_res

incep_res_features = pickle.load(open('incep_res_features', 'rb')) # run this instead of the line below
incep_res_features = extract_bottleneck(path=target_path, model=InceptionResNetV2,
                                        preprocesser=preprocess_input_incep_res, filename='incep_res_features')
incep_res_train, incep_res_test = train_test_split(incep_res_features, random_state=42, shuffle=True)


##### Creating a branch for each set bottleneck features #####

def branch(input_shape=None, augment=False, reg=0.0):
    size = int(input_shape[2] / 4)
    x_input = Input(shape=input_shape)

    if augment == True:
        x = data_augmentation(x_input)
        x = GlobalAveragePooling2D()(x)
    else:
        x = GlobalAveragePooling2D()(x_input)

    x = Dense(size, activation='relu', kernel_regularizer=l2(reg), kernel_initializer=he_initializer)(x)
    x = BatchNormalization()(x)
    return x, x_input


resnetv2_branch, resnetv2_input = branch(resnetv2_train.shape[1:])
xcep_branch, xcep_input = branch(xcep_train.shape[1:])
eff_branch, eff_input = branch(efficient_train.shape[1:])
incep_res_branch, incep_res_input = branch(incep_res_train.shape[1:])


##### Concatenating all the outputs and compiling model #####

def concat_model_top(branches=[], drop=0.3, reg=0.0):
    concatenated_branch = Concatenate()(branches)
    x = Dropout(drop)(concatenated_branch)
    x = Dense(512, activation='relu', kernel_regularizer=l2(reg), kernel_initializer=he_initializer)(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x_out = Dense(133, activation='softmax', kernel_regularizer=l2(reg), kernel_initializer=he_initializer)(x)
    return x_out


x_out = concat_model_top([resnetv2_branch, xcep_branch, eff_branch, incep_res_branch], drop=0.8)

model = Model(inputs=[resnetv2_input, xcep_input, eff_input, incep_res_input], outputs=[x_out])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = callbacks_funcv2('checkpoints_4model_drop80')

##### Training and results #####

history = model.fit(
    [resnetv2_train, xcep_train, eff_train, incep_res_train], y_train,
    validation_data=([resnetv2_test, xcep_test, eff_test, incep_res_test], y_test),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

history1 = pd.read_csv('checkpoints_4model_drop30/log.csv')
history2 = pd.read_csv('checkpoints_4model_drop70/log.csv')
history3 = pd.read_csv('checkpoints_4model_drop70_reg0001/log.csv')
history4 = pd.read_csv('checkpoints_4model_drop80/log.csv')

graph_history(history1, history2, history3, history4, main_title='Model Comparison', axes_titles=['dropout rate 0.3',
                                                                                                  'dropout rate 0.7',
                                                                                                  'dropout rate 0.7,'
                                                                                                  '\nlambda = 0.0001',
                                                                                                  'dropout rate 0.8'],
              xmax=50)

##### Reconstructing #####

from tensorflow.keras.models import load_model

bottleneck_model = load_model('checkpoints_4model_drop70/epoch_25-val_acc_0.89.hdf5')

resnetv2_model = ResNet152V2(include_top=False, weights="imagenet", input_shape=(200, 200, 3))
xception_model = Xception(include_top=False, weights="imagenet", input_shape=(200, 200, 3))
efficient_model = EfficientNetB7(include_top=False, weights="imagenet", input_shape=(200, 200, 3))
incep_res_model = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(200, 200, 3))

x_input = Input(shape=(200, 200, 3))

pre_res = preprocess_input_resnet152(x_input)
pre_xcep = preprocess_input_xception(x_input)
pre_eff = preprocess_input_efficient(x_input)
pre_incep_res = preprocess_input_incep_res(x_input)

res_output = resnetv2_model(pre_res)
xcep_output = xception_model(pre_xcep)
eff_output = efficient_model(pre_eff)
incep_output = incep_res_model(pre_incep_res)

whole_output = bottleneck_model([res_output, xcep_output, eff_output, incep_output])

whole_model = Model(inputs=x_input, outputs=whole_output)
whole_model.compile(optimizer='adam')
whole_model.save('whole_model')


# /------------------------
# Section 6: Jessie and Friends
# /------------------------

def breed_classifier(path_to_img, model):
    original_img = load_img(path_to_img)
    resized_img = load_img(path_to_img, target_size=(200, 200))
    array_img = np.asarray(resized_img).reshape(1, 200, 200, 3)

    predictions = model.predict(array_img).reshape(133)
    top_3_ind = predictions.argsort()[-3:][::-1]  # top 3

    percentage = predictions[top_3_ind]  # top 3's percentage
    percentage_rounded = [str(np.round(x, 2)) + '%' for x in percentage * 100]

    prediction_labels = np.array(labels_str)[top_3_ind]
    prediction_dict = dict(zip(prediction_labels, percentage_rounded))

    return prediction_dict, original_img, top_3_ind


import random

def prediction_graph(friend='Jessie'):
    friend = friend.capitalize()  # capitalizing to put respect in your friend's names

    for i in os.listdir(friend):
        path = friend + '/' + i
        prediction, original_img, top_3_ind = breed_classifier(path, model)

        plt.figure(figsize=(11, 11))

        ax1 = plt.subplot2grid((4, 4), (0, 3))  # actual
        ax2 = plt.subplot2grid((4, 4), (1, 3))  # guess 1
        ax3 = plt.subplot2grid((4, 4), (2, 3))  # guess 2
        ax4 = plt.subplot2grid((4, 4), (3, 3))  # guess 3

        ax0 = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=4)

        top_3_path = ['dogImages - combined/' + os.listdir('dogImages - combined')[i] for i in top_3_ind]
        top_3_graph = [i + '/' + random.choice(os.listdir(i)) for i in top_3_path]

        ax0.imshow(original_img)
        ax0.set_title(friend, fontsize=10)
        ax0.axis('off')

        if friend == "Jessie":
            actual_breed_path = 'dogImages - combined/104.Miniature_schnauzer'
            ax1.set_title('Actual Breed: Miniature Schnauzer', fontsize=10)

        elif friend == "Merry":
            actual_breed_path = 'dogImages - combined/124.Poodle'
            ax1.set_title('Actual Breed: (Miniature) Poodle', fontsize=10)

        elif friend == "Butter":
            actual_breed_path = 'dogImages - combined/045.Cardigan_welsh_corgi'
            ax1.set_title('Actual Breed: Welsh Corgi', fontsize=10)

        elif friend == "Nala":
            actual_breed_path = 'dogImages - combined/071.German_shepherd_dog'
            ax1.set_title('Actual Breed: (White) German Shephard', fontsize=10)

        actual_random_path = actual_breed_path + '/' + random.choice(os.listdir(actual_breed_path))
        ax1.imshow(load_img(actual_random_path))  # actual
        ax1.axis('off')

        ax2.imshow(load_img(top_3_graph[0]))  # guesses

        clean_title = [': '.join(list(i)) for i in list(prediction.items())]
        ax2.set_title(clean_title[0], fontsize=10)
        ax2.axis('off')

        ax3.imshow(load_img(top_3_graph[1]))  # guesses
        ax3.set_title(clean_title[1], fontsize=10)
        ax3.axis('off')

        ax4.imshow(load_img(top_3_graph[2]))  # guesses
        ax4.set_title(clean_title[2], fontsize=10)
        ax4.axis('off')


from keras.models import load_model

model = load_model('whole_model')
model.summary()

##### Jessie #####
prediction_graph("Jessie")

##### Butter #####
prediction_graph("Butter")

##### Merry #####
prediction_graph("Merry")

##### Nala #####
prediction_graph("Nala")



# /------------------------
# Conclustion
# /------------------------

jessie_path = 'Jessie/jessie4.jpg'

original = load_img(jessie_path)
resize = load_img(jessie_path, target_size=(200, 200))

fig, axs = plt.subplots(1, 2)
fig.set_size_inches(10, 10)

axs[1].imshow(original)
axs[0].imshow(resize)

[axi.set_axis_off() for axi in axs.ravel()]