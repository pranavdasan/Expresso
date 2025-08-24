import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import cv2
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Use directly uploaded zip file
import zipfile
import os

# Get the uploaded zip filename
zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
if zip_files:
    dataset_zip = zip_files[0]  # Use first zip file found
    print(f"Using zip file: {dataset_zip}")
else:
    print("No zip file found. Make sure you uploaded a zip file.")
    dataset_zip = "your-dataset.zip"  # fallback

# Extract the zip file
with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall('Facial Expressions Dataset/')

print(f"Extracted {dataset_zip} to 'Facial Expressions Dataset/' folder")

# Paths to datasets
trainPath = "Facial Expressions Dataset/train"
testPath = "Facial Expressions Dataset/test"

# Arrays for data
X_train = []
y_train = []
X_test = []
y_test = []


# Load the data into the arrays
print("Loading training data...") # 28709 train images
folderList = os.listdir(trainPath)
folderList.sort()

for i, category in enumerate(folderList):
    files = os.listdir(trainPath+"/"+category)
    for file in files:
        # print(category+"/"+file)
        img = cv2.imread(trainPath+"/"+category+"/{0}".format(file), 0) # Set to grayscale mode
        X_train.append(img)
        y_train.append(i)


print("Loading testing data...")
folderList = os.listdir(testPath)
folderList.sort()

for i, category in enumerate(folderList):
    files = os.listdir(testPath+"/"+category)
    for file in files:
        # print(category+"/"+file)
        img = cv2.imread(testPath+"/"+category+"/{0}".format(file), 0) # Set to grayscale mode
        X_test.append(img)
        y_test.append(i)


# Convert the data to numpy
X_train = np.array(X_train, 'float32')
y_train = np.array(y_train, 'float32')
X_test = np.array(X_test, 'float32')
y_test = np.array(y_test, 'float32')


# Normalize the data: All the pixels will turn to either 0 or 1
X_train = X_train / 255.0
X_test = X_test / 255.0


# Add another dimension to the data to (28709, 48, 48, 1)
# For training data
numOfImages = X_train.shape[0]
X_train = X_train.reshape(numOfImages, 48, 48, 1) # Added another dimension for grayscale image

# For test data
numOfImages = X_test.shape[0]
X_test = X_test.reshape(numOfImages, 48, 48, 1)


# Convert the labels into categories
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)


'''
Build the model

    Conv2D: Helps find features like edges and textures by applying filters to it
        - filters: Currenly have it in increasing size so the model will detect more complex features as the image passes through the layers
        - kernel_size: Determines the area of the input image that a single filter will consider at a time, small means faster but less complex
        - padding: Makes sure the image isn't becoming smaller as it goes through the layers
    
    MaxPooling: Takes the best/maximum features from the feature maps, reducing the total amount of parameters and computations making the model
                overall faster and less prone to overfitting
        - pool_size: Window the model will take the "best" feature from will be 2x2 pixels
        - strides: Moves the window by 2 pixels at a time

    Flatten: Takes the 3D output of the layers behind it and compacts it to a 1D vector for the upcoming layers

    Dense: A layer of fully connected neurons
        - L2 Regularization:  Penalizes large weights, forcing the model to use smaller, more distributed weights

    Dropout: Deactivates half of the neurons it chooses, this is to prevent overfitting
'''

input_shape = X_train.shape[1:]

model = Sequential()
model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(4096, activation="relu", kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu", kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(7, activation="softmax")) # 7 Neurons for the 7 categories, softmax helps in classification

print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

