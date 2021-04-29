#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 20:10:12 2021

@author: alessiosavi
"""

# %%
import shutil
import datetime
import pickle
from mtcnn import MTCNN
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
# import insightface
import dlib
from sklearn.preprocessing import LabelEncoder
import cv2
from tensorflow.keras import layers
from os.path import join as pjoin
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# %%
basedir = "lfw"
train_size = 0.8
face_detector = MTCNN()

pose_predictor = dlib.shape_predictor(
    'dlib-face-recognition/models/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1(
    'dlib-face-recognition/models/dlib_face_recognition_resnet_model_v1.dat')

# %%
# extract a single face from a given photograph


def extract_faces(filename):
    faces = []
    # load image from file
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect faces in the image
    results = face_detector.detect_faces(image)
    for result in results:
        x, y, width, height = result["box"]
        faces.append(dlib.rectangle(x, y, x+width, y+height))
    return image, faces


def encodings(img, face_locations, pose_predictor, face_encoder):
    predictors = [pose_predictor(img, face_location)
                  for face_location in face_locations]
    return [np.array(face_encoder.compute_face_descriptor(img, predictor, 1)) for predictor in predictors]


# %%
X_train = []
y_train = []
X_test = []
y_test = []


def load_dataset_dlib():
    # Iterate every folder of dataset dir
    # for person in tqdm(os.listdir(basedir), desc=person, miniters=1):
    for person in tqdm(os.listdir(basedir)):
        # Path related to all photos of a person
        person_path = pjoin(basedir, person)
        # All photos related to a person
        person_photos = os.listdir(person_path)
        # Avoid to manage person that have less than 10 photo
        if len(person_photos) < 10:
            print("Skipping {} due to {} images ...",
                  person, len(person_photos))
            continue
        # Load the TRAIN dataset
        for photo_path in person_photos[0: int(len(person_photos) * train_size)]:
            img, face_locations = extract_faces(pjoin(person_path, photo_path))
            if len(face_locations) != 1:   # Avoid to manage photo where there are more than one face
                print("Skipping {} due to {} faces recognized".format(
                    pjoin(person_path, photo_path), len(face_locations)))
                continue
            face_encodings = encodings(
                img, face_locations, pose_predictor, face_encoder)
            X_train.append(face_encodings[0])
            y_train.append(person)
        # Load the TEST dataset
        for photo_path in person_photos[int(len(person_photos) * train_size):]:
            img, face_locations = extract_faces(pjoin(person_path, photo_path))
            if len(face_locations) != 1:   # Avoid to manage photo where there are more than one face
                print("Skipping {} due to {} faces recognized".format(
                    pjoin(person_path, photo_path), len(face_locations)))
                continue
            face_encodings = encodings(
                img, face_locations, pose_predictor, face_encoder)
            X_test.append(face_encodings[0])
            y_test.append(person)
    with open('dataset_dlib.pkl', 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    return X_train, X_test, y_train, y_test

# %%


# X_train, X_test, y_train, y_test = load_dataset_dlib()

with open('dataset_dlib.pkl', 'rb') as f:
    (X_train, X_test, y_train, y_test) = pickle.load(f)

# %%
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
# %% CREATE MODEL
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)


model = tf.keras.Sequential(
    [
        layers.Dense(128, activation="relu", input_shape=(train_x.shape[1],)),
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
model.fit(train_x, y_train_label, validation_data=(
    test_x, y_test_label),  epochs=50, callbacks=[tensorboard_callback])

# %% EVALUATE MODEL
loss, acc = model.evaluate(test_x, y_test_label)
print("Accuracy", acc)

# %%

print("Predict an image")
image_to_predict = test_x[90]
predictions = model.predict(image_to_predict.reshape(1, 128))[0]
label_predicted = np.argmax(predictions)
confidence = predictions[label_predicted]
predicted_name = label_encoder.inverse_transform([label_predicted])[0]
print("Name predicted: {} | Confidence: {}".format(predicted_name, confidence))
