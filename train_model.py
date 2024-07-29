import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit

LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

# Initialize the data and labels lists
data = []
labels = []

# Loop over the input images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the letter image to fit within a 20x20 pixel box
    image = resize_to_fit(image, 20, 20)

    # Add a third channel dimension to the image (Keras requirement)
    image = np.expand_dims(image, axis=2)

    # Get the label (the letter) from the folder name
    label = image_file.split(os.path.sep)[-2]

    # Append the processed image and its label to the respective lists
    data.append(image)
    labels.append(label)

# Scale the pixel intensities to the range [0, 1] for better training
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split the data into training and testing sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# Convert the labels to one-hot encodings for Keras
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the label binarizer to disk to map predictions to their corresponding labels
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Build the neural network model
model = Sequential()

# Add the first convolutional layer followed by max pooling
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Add the second convolutional layer followed by max pooling
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Add a fully connected hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# Add the output layer with one node per possible letter/number
model.add(Dense(32, activation="softmax"))

# Compile the model with categorical crossentropy loss, Adam optimizer, and accuracy metrics
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model on the training data
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)

# Save the trained model to disk
model.save(MODEL_FILENAME)
