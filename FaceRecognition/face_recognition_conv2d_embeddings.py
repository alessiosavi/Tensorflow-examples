# %% IMPORT
import face_recognition
from sklearn.preprocessing import LabelEncoder
import os
from typing import List
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
from tqdm import tqdm
import face_recognition
# %% LOAD IMAGE CLASSIFIER

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
# %% DEFINE VAR

basedir = "lfw"
w, h = 224, 224
train_size = 0.8
person = ''

# %% FILTER FACE FROM IMAGE METHOD

# Retrieve the 128 points of the face


def get_face_landmark(img: np.array):
    faceBlob = cv2.dnn.blobFromImage(
        img, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    return vec.ravel()

# OPENCV DNN


def filter_single_face_dnn(img_path: str) -> List[np.ndarray]:
    faces = []
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, dsize=(
        h, w), interpolation=cv2.INTER_LANCZOS4), 1.0, (h, w), (104.0, 117.0, 123.0))
    detector.setInput(blob)
    faces3 = detector.forward()
    for i in range(faces3.shape[2]):
        confidence = faces3[0, 0, i, 2]
        if confidence > 0.9:
            box = faces3[0, 0, i, 3:7] * \
                np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            crop_img = img[y:y1, x:x1]
            try:
                faces.append(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            except Exception:
                print("Unable to load "+img_path)
    return faces

# %% LOAD X and y


X_train = []
y_train = []
X_test = []
y_test = []
# Iterate every folder of dataset dir
for person in tqdm(os.listdir(basedir), desc=person):
    # Path related to all photos of a person
    person_path = os.path.join(basedir, person)
    # All photos related to a person
    person_photos = os.listdir(person_path)
    # Avoid to manage person that have less than 10 photo
    if len(person_photos) < 10:
        print("Skipping {} due to {} images ...", person, len(person_photos))
        continue
    # Load the TRAIN dataset
    for photo_path in person_photos[0: int(len(person_photos) * train_size)]:
        photos = filter_single_face_dnn(os.path.join(person_path, photo_path))
        # Avoid to manage photo where there are more than one face
        if len(photos) != 1:
            # print("Skipping {} due to {} faces recognized".format(os.path.join(person_path, photo_path), len(photos)))
            continue
        X_train.append(get_face_landmark(photos[0]))
        y_train.append(person)
    # Load the TEST dataset
    for photo_path in person_photos[int(len(person_photos) * train_size):]:
        photos = filter_single_face_dnn(os.path.join(person_path, photo_path))
        if len(photos) != 1:
            # print("Skipping {} due to {} faces recognized".format(os.path.join(person_path, photo_path), len(photos)))
            continue
        X_test.append(get_face_landmark(photos[0]))
        y_test.append(person)


# %% ENCODE LABEL
# Transform name from string to int
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)
label_encoder = label_encoder.fit(y_test)
y_train_label = label_encoder.transform(y_train)
y_test_label = label_encoder.transform(y_test)

# %% CAST DATASET
train_x = np.array(X_train)
test_x = np.array(X_test)
# %% CREATE MODEL

model = tf.keras.Sequential(
    [
        layers.Dense(128, activation="relu", input_shape=(train_x.shape[1],)),
        # layers.Dense(512, activation="relu"),
        # layers.Dense(256, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(32, activation="relu"),
        # layers.Dense(32, activation="relu", input_shape=(train_x.shape[1],)),
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
    test_x, y_test_label),  epochs=200)

# %% EVALUATE MODEL
loss, acc = model.evaluate(test_x, y_test_label)
print("Accuracy", acc)

# %% SIMPLE PREDICTION

print("Predict an image")
image_to_predict = test_x[4:5]
predictions = model.predict(image_to_predict).ravel()
label_predicted = np.argmax(predictions)
confidence = predictions[label_predicted]
predicted_name = label_encoder.inverse_transform([label_predicted])[0]
print("Name predicted: {} | Confidence: {}".format(predicted_name, confidence))
plt.imshow(image_to_predict[0])


# %%

# %%

X_train = []
y_train = []
X_test = []
y_test = []
# Iterate every folder of dataset dir
for person in tqdm(os.listdir(basedir), desc=person, miniters=1):
    # Path related to all photos of a person
    person_path = os.path.join(basedir, person)
    # All photos related to a person
    person_photos = os.listdir(person_path)
    # Avoid to manage person that have less than 10 photo
    if len(person_photos) < 10:
        print("Skipping {} due to {} images ...", person, len(person_photos))
        continue
    # Load the TRAIN dataset
    for photo_path in person_photos[0: int(len(person_photos) * train_size)]:
        photo = face_recognition.load_image_file(
            os.path.join(person_path, photo_path))
        encodings = face_recognition.face_encodings(photo, model='large')
        # Avoid to manage photo where there are more than one face
        if len(encodings) != 1:
            print("Skipping {} due to {} faces recognized".format(
                os.path.join(person_path, photo_path), len(encodings)))
            continue
        X_train.append(encodings[0])
        y_train.append(person)
    # Load the TEST dataset
    for photo_path in person_photos[int(len(person_photos) * train_size):]:
        photo = face_recognition.load_image_file(
            os.path.join(person_path, photo_path))
        encodings = face_recognition.face_encodings(photo, model='large')
        if len(encodings) != 1:
            print("Skipping {} due to {} faces recognized".format(
                os.path.join(person_path, photo_path), len(encodings)))
            continue
        X_test.append(encodings[0])
        y_test.append(person)

# %% ENCODE LABEL
# Transform name from string to int
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)
label_encoder = label_encoder.fit(y_test)
y_train_label = label_encoder.transform(y_train)
y_test_label = label_encoder.transform(y_test)


# %% CAST DATASET
train_x = np.array(X_train)
test_x = np.array(X_test)
# %% CREATE MODEL

model = tf.keras.Sequential(
    [
        layers.Dense(16, activation="relu", input_shape=(train_x.shape[1],)),
        # layers.Dropout(0.1),
        # layers.Dense(16, activation="relu"),
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
    test_x, y_test_label), batch_size=8, epochs=100)


# %% EVALUATE MODEL
loss, acc = model.evaluate(test_x, y_test_label)
print("Accuracy", acc)
