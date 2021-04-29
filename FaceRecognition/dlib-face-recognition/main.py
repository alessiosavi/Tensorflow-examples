#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 20:10:12 2021

@author: alessiosavi
"""

# %%
import matplotlib.pyplot as plt
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
import tensorflow as tf
import cv2
from tensorflow.keras import layers
import datetime
from os.path import join as pjoin
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# %%
basedir = "/opt/SP/workspace/JupyterLab/Tensorflow-Certification/FaceRecognition/lfw"
person = ''
train_size = 0.8
face_detector = MTCNN()

# model = insightface.model_zoo.get_model('arcface_r100_v1')
# model.prepare(ctx_id=0)
pose_predictor = dlib.shape_predictor(
    'dlib-face-recognition/models/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1(
    'dlib-face-recognition/models/dlib_face_recognition_resnet_model_v1.dat')

# %%
# extract a single face from a given photograph


def extract_faces(filename, required_size=(250, 250), mode="DLIB"):
    faces = []
    # load image from file
    if mode == "DLIB":
        image = cv2.imread(filename)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # detect faces in the image
        results = face_detector.detect_faces(image)
    else:
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = np.asarray(image)
        # detect faces in the image
        results = face_detector.detect_faces(pixels)
    for result in results:
        x, y, width, height = result["box"]
        if mode == "DLIB":
            faces.append(dlib.rectangle(x, y, x+width, y+height))
        else:
            x1, y1 = abs(x), abs(y)
            x2, y2 = x1 + width, y1 + height
            # extract the face
            # convert to array
            pixels = np.asarray(image)
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = np.asarray(image)
            faces.append(face_array)
    return image, faces


def get_encodings(filename):
    img, face_locations = extract_faces(filename, mode="DLIB")
    face_encodings = encodings(
        img, face_locations, pose_predictor, face_encoder)
    return face_encodings


def encodings(img, face_locations, pose_predictor, face_encoder):
    predictors = [pose_predictor(img, face_location)
                  for face_location in face_locations]
    return [np.array(face_encoder.compute_face_descriptor(img, predictor, 1)) for predictor in predictors]


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


# x = extract_faces("/opt/SP/workspace/JupyterLab/Tensorflow-Certification/FaceRecognition/lfw/Abdullah_Gul/Abdullah_Gul_0001.jpg")
# emb = model.get_embedding(x[0])
# %%
X_train = []
y_train = []
X_test = []
y_test = []


# def load_dataset_insightface():
#     # Iterate every folder of dataset dir
#     # for person in tqdm(os.listdir(basedir), desc=person, miniters=1):
#     for person in tqdm(os.listdir(basedir)):
#         # Path related to all photos of a person
#         person_path = pjoin(basedir, person)
#         # All photos related to a person
#         person_photos = os.listdir(person_path)
#         # Avoid to manage person that have less than 10 photo
#         if len(person_photos) < 10:
#             print("Skipping {} due to {} images ...",
#                   person, len(person_photos))
#             continue
#         # Load the TRAIN dataset
#         for photo_path in person_photos[0: int(len(person_photos) * train_size)]:
#             photos = extract_faces(pjoin(person_path, photo_path))
#             if len(photos) != 1:   # Avoid to manage photo where there are more than one face
#                 print("Skipping {} due to {} faces recognized".format(
#                     pjoin(person_path, photo_path), len(photos)))
#                 continue
#             emb = model.get_embedding(photos[0])[0]
#             X_train.append(emb)
#             y_train.append(person)
#         # Load the TEST dataset
#         for photo_path in person_photos[int(len(person_photos) * train_size):]:
#             photos = extract_faces(pjoin(person_path, photo_path))
#             if len(photos) != 1:   # Avoid to manage photo where there are more than one face
#                 print("Skipping {} due to {} faces recognized".format(
#                     pjoin(person_path, photo_path), len(photos)))
#                 continue
#             emb = model.get_embedding(photos[0])[0]
#             X_test.append(emb)
#             y_test.append(person)
#     with open('dataset_insight.pkl', 'wb') as f:
#         pickle.dump((X_train, X_test, y_train, y_test), f)

import shutil
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
            try:
                shutil.rmtree(person_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
            print("Skipping {} due to {} images ...",
                  person, len(person_photos))
            continue
        # Load the TRAIN dataset
        for photo_path in person_photos[0: int(len(person_photos) * train_size)]:
            img, face_locations = extract_faces(
                pjoin(person_path, photo_path), mode="DLIB")
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
            img, face_locations = extract_faces(
                pjoin(person_path, photo_path), mode="DLIB")
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
        # layers.Dense(512, activation="relu"),
        # layers.Dense(64, activation="relu"),
        # layers.Dropout(0.1),
        # layers.Dense(32, activation="relu"),
        # layers.Dense(128, activation="relu"),
        # layers.Dropout(0.1),
        # layers.Dense(256, activation="relu"),
        # layers.Dense(32, activation="relu", input_shape=(train_x.shape[1],)),
        layers.Dense(len(set(y_test_label)), activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    # loss = "categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()


# %% FIT MODEL
model.fit(train_x, y_train_label, validation_data=(
    test_x, y_test_label),  epochs=500, callbacks=[tensorboard_callback])

# %% EVALUATE MODEL
loss, acc = model.evaluate(test_x, y_test_label)
print("Accuracy", acc)

# %%

# Create a new dataset that contains only the cropped image of the people
for person in tqdm(os.listdir(basedir)):
    person_path = pjoin(basedir, person)
    if len(os.listdir(person_path)) < 10:
        continue
    for person_photo in os.listdir(person_path):
        img, face_locations = extract_faces(
            pjoin(person_path, person_photo), mode="DLIB")
        if len(face_locations) == 1:
            # print("File {} contains {} face!".format(pjoin(person_path, person_photo), len(face_locations)))
            for face in face_locations:
                face_encodings = encodings(
                    img, [face], pose_predictor, face_encoder)
                predictions = model.predict(
                    face_encodings[0].reshape(1, 128))[0]
                label_predicted = np.argmax(predictions)
                confidence = predictions[label_predicted]
                predicted_name = label_encoder.inverse_transform([label_predicted])[
                    0]
                if predicted_name != person:
                    print("Name predicted: {} | Real Name: {} Confidence: {}".format(
                        predicted_name, person, confidence))


# %%
a = "/opt/SP/workspace/JupyterLab/Tensorflow-Certification/FaceRecognition/lfw/George_HW_Bush/George_HW_Bush_0001.jpg"
b = get_encodings(a)

# %%

print("Predict an image")
image_to_predict = test_x[90]
predictions = model.predict(image_to_predict.reshape(1, 128))[0]
label_predicted = np.argmax(predictions)
confidence = predictions[label_predicted]
predicted_name = label_encoder.inverse_transform([label_predicted])[0]
print("Name predicted: {} | Confidence: {}".format(predicted_name, confidence))


# %%
def convert_and_trim_bb(image, rect):
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return (startX, startY, w, h)

# %%

# %%


# %%

# %%


# %%

# %%


# %%

# %%


# %%

# %%


# %%

# %%


# %%

# %%
