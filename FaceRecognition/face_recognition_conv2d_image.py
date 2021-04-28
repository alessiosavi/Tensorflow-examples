# %% IMPORT
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
# %% LOAD IMAGE CLASSIFIER
face_cascade = cv2.CascadeClassifier(
    "/opt/SP/software/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
)

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = "dnn_models/deploy.prototxt"
modelPath = "dnn_models/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# load our serialized face embedding model from disk
# print("[INFO] loading face recognizer...")
# embedder = cv2.dnn.readNetFromTorch("dnn_models/openface.nn4.small2.v1.t7")
# embedder.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#%% DEFINE VAR

basedir = "lfw"
w, h = 224, 224

train_size = 0.8
X_train = []
y_train = []
X_test = []
y_test = []
person = ''

#%% FILTER FACE FROM IMAGE METHOD
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
            crop_img = a[y: y + h, x: x + w]
            faces.append(crop_img)
    return faces


# OPENCV DNN
def filter_single_face_dnn(img_path: str) -> List[np.ndarray]:
    faces = []
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_LANCZOS4), 1.0, (h, w), (104.0, 117.0, 123.0) )
    # blob = cv2.dnn.blobFromImage(img, 1.0/255, (h, w))
    detector.setInput(blob)
    faces3 = detector.forward()
    for i in range(faces3.shape[2]):
        confidence = faces3[0, 0, i, 2]
        if confidence > 0.9:
            box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            crop_img = img[y:y1, x:x1]
            try:
                faces.append(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            except Exception:
                print("Unable to load "+img_path)
    return faces

#%% LOAD X and y

# Iterate every folder of dataset dir
for person in tqdm(os.listdir(basedir)[0:20], desc=person):
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
        X_train.append(cv2.resize(photos[0], dsize=(h, w), interpolation=cv2.INTER_LANCZOS4))
        y_train.append(person)
    # Load the TEST dataset
    for photo_path in person_photos[int(len(person_photos) * train_size):]:
        photos = filter_single_face_dnn(os.path.join(person_path, photo_path))
        if len(photos) != 1:
            # print("Skipping {} due to {} faces recognized".format(os.path.join(person_path, photo_path), len(photos)))
            continue
        X_test.append(cv2.resize(photos[0], dsize=(h, w), interpolation=cv2.INTER_LANCZOS4))
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
        layers.experimental.preprocessing.Rescaling(1.0 / 255, input_shape=(h, w, 3)),
        # layers.experimental.preprocessing.RandomFlip("horizontal"),
        # layers.experimental.preprocessing.RandomRotation(0.2),
        layers.Conv2D(128, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        # tf.keras.applications.VGG16(weights=None,classes=len(set(y_test_label)),input_shape=((h, w, 3))),
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
model.fit(train_x, y_train_label, validation_data=(test_x, y_test_label), epochs=15)

# %% EVALUATE MODEL
loss, acc = model.evaluate(test_x, y_test_label)
print("Accuracy", acc)

#%% SIMPLE PREDICTION

print("Predict an image")
image_to_predict = test_x[4:5]
predictions = model.predict(image_to_predict).ravel()
label_predicted = np.argmax(predictions)
confidence = predictions[label_predicted]
predicted_name = label_encoder.inverse_transform([label_predicted])[0]
print("Name predicted: {} | Confidence: {}".format(predicted_name,confidence))
plt.imshow(image_to_predict[0])


