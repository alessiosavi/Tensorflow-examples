#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 20:10:12 2021

@author: alessiosavi
"""

# %%
import datetime
import pickle
from mtcnn import MTCNN
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import dlib
from sklearn.preprocessing import LabelEncoder
import cv2
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from os.path import join as pjoin
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# %%
basedir = "lfw2"
person = ''
train_size = 0.8
face_detector = MTCNN()
img_height, img_width = 250, 250
# %%


def extract_faces(filename="", image=None):
    faces = []
    # load image from file
    if image is None:
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect faces in the image
    results = face_detector.detect_faces(image)

    for result in results:
        x, y, width, height = result["box"]
        faces.append(dlib.rectangle(x, y, x+width, y+height))
    return image, faces


def resize_image(img, h, w):
    return cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_LANCZOS4)


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = abs(rect.left())
    y = abs(rect.top())
    w = abs(rect.right() - x)
    h = abs(rect.bottom() - y)

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)
# %%


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


X_train = []
y_train = []
X_test = []
y_test = []


def image_augmentation(basedir):
    # Iterate every folder of dataset dir
    for person in tqdm(os.listdir(basedir)):
        # Path related to all photos of a person
        person_path = pjoin(basedir, person)
        # All photos related to a person
        person_photos = os.listdir(person_path)

        # Avoid to manage person that have less than 10 photo
        if len(person_photos) < 10:
            try:
                shutil.rmtree(person_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
            print("Removing {} due to {} images ...",
                  person, len(person_photos))
            continue
        # Load the TRAIN dataset
        for photo_path in person_photos[0: int(len(person_photos) * train_size)]:
            img, face_locations = extract_faces(pjoin(person_path, photo_path))
            if len(face_locations) != 1:   # Avoid to manage photo where there are more than one face
                print("Skipping {} due to {} faces recognized".format(
                    pjoin(person_path, photo_path), len(face_locations)))
                continue
            x, y, w, h = rect_to_bb(face_locations[0])
            crop_img = img[y:y+h, x:x+w]
            X_train.append(crop_img)
            y_train.append(person)
        # Load the TEST dataset
        for photo_path in person_photos[int(len(person_photos) * train_size):]:
            img, face_locations = extract_faces(pjoin(person_path, photo_path))
            if len(face_locations) != 1:   # Avoid to manage photo where there are more than one face
                print("Skipping {} due to {} faces recognized".format(
                    pjoin(person_path, photo_path), len(face_locations)))
                continue
            X_test.append(img)
            y_test.append(person)
    with open('dataset_dlib_face.pkl', 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = image_augmentation(basedir)

# with open('dataset_dlib_face.pkl', 'rb') as f:
#     (X_train, X_test, y_train, y_test) = pickle.load(f)

assert len(X_train) == len(y_train) and len(y_train) > 0
assert len(X_test) == len(y_test) and len(y_test) > 0
# %%

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)
label_encoder = label_encoder.fit(y_test)
y_train_label = label_encoder.transform(y_train)
y_test_label = label_encoder.transform(y_test)

# %% CAST DATASET
train_x = np.array(X_train)
test_x = np.array(X_test)
# test_x = test_x.reshape(test_x.shape[0],test_x.shape[2])
# train_x = train_x.reshape(train_x.shape[0],train_x.shape[2])
print(train_x.shape, test_x.shape, y_train_label.shape, y_test_label.shape)


# %%
datagen.fit(train_x)

# %% CREATE MODEL
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)


model = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(
            1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(set(y_test_label)), activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary()


# %% FIT MODEL
model.fit(datagen.flow(train_x, y_train_label), validation_data=(
    test_x, y_test_label),  epochs=10, callbacks=[tensorboard_callback])

# %% EVALUATE MODEL
loss, acc = model.evaluate(test_x, y_test_label)
print("Accuracy", acc)
