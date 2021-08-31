#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
from tqdm import tqdm


# In[3]:


face_cascade = cv2.CascadeClassifier(
    "/opt/SP/software/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
)


# In[4]:


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = "dnn_models/deploy.prototxt"
modelPath = "dnn_models/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("dnn_models/openface.nn4.small2.v1.t7")
embedder.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# In[39]:


def filter_single_face(img_path: str) -> List[np.ndarray]:
    faces = []
    # Read the image
    a = cv2.imread(img_path)
    # Change color
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    # box_points: position (x,y,w,h) of the faces
    # confidence: robustness of the prediction
    box_points, _, confidence = face_cascade.detectMultiScale3(
        gray, minNeighbors=50, outputRejectLevels=True
    )
    for i in range(len(confidence)):
        # Filter for robust predictin
        if confidence[i] > 2:
            x, y, w, h = (
                box_points[i][0],
                box_points[i][1],
                box_points[i][2],
                box_points[i][3],
            )
            # Only save the face, anything else
            crop_img = a[y : y + h, x : x + w]
            faces.append(crop_img)
    return faces


x = filter_single_face("/home/alessiosavi/Downloads/a.jpg")
len(x)


# In[76]:


# OPENCV DNN
def filter_single_face_dnn(img_path: str) -> List[np.ndarray]:
    faces = []
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
    )
    detector.setInput(blob)
    faces3 = detector.forward()
    for i in range(faces3.shape[2]):
        confidence = faces3[0, 0, i, 2]
        if confidence > 0.9:
            box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            crop_img = img[y:y1, x:x1]
            faces.append(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    return faces


x = filter_single_face_dnn("/home/alessiosavi/Downloads/a.jpg")
len(x)


# In[41]:


basedir = "lfw"


# In[79]:


filter_single_face_dnn(os.path.join(person_path, photo_path))


# In[78]:


os.path.join(person_path, photo_path)


# In[77]:


train_size = 0.8
X_train = []
y_train = []
for person in tqdm(os.listdir(basedir), desc=person):
    person_path = os.path.join(basedir, person)
    person_photo = os.listdir(person_path)
    if len(person_photo) < 10:
        print("Skipping {} due to {} images ...",person_photo,len(person_photo))
        continue
    for photo_path in person_photo[0 : int(len(person_photo) * train_size)]:
        photos = filter_single_face_dnn(os.path.join(person_path, photo_path))
        if len(photos) > 1:
            print(
                "Skipping {} due to {} faces recognized".format(
                    os.path.join(person_path, photo_path), len(photos)
                )
            )
            continue
        X_train.append(photos[0])
        y_train.append(person)
#     for photo in person_photo[int(len(person_photo)*train_size):]:

