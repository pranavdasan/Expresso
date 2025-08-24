import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import cv2
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Paths to datasets
trainPath = "C:/Pranav Items/Expresso/Facial Expressions Dataset/train"
testPath = "C:/Pranav Items/Expresso/Facial Expressions Dataset/test"

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


# img1 = X_train[0]

# cv2.imshow("img1", img1)
# cv2.waitKey(0)

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

    Dropout: Deactivates half of the neurons it chooses, this is to prevent overfitting
'''

input_shape = X_train.shape[1:]

model = Sequential()
model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dense(7, activation="softmax")) # 7 Neurons for the 7 categories, softmax helps in classification

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
batch = 32
epochs = 30

stepsPerEpoch = np.ceil(len(X_train)/batch)
validationSteps = np.ceil(len(X_test)/batch) 

stopEarly = EarlyStopping(monitor='val_accuracy', patience=5)

history = model.fit(X_train,
                    y_train,
                    batch_size=batch,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    shuffle=True,
                    callbacks=[stopEarly])


# Show the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, acc, 'r', label="Train accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation accuracy")
plt.xlabel('Epoch')
plt.xlabel('Accuracy')
plt.title("Training and Validation Accuracy")
plt.legend(loc='lower right')
plt.show()

plt.plot(epochs, acc, 'r', label="Train loss")
plt.plot(epochs, val_acc, 'b', label="Validation loss")
plt.xlabel('Epoch')
plt.xlabel('Loss')
plt.title("Training and Validation Loss")
plt.legend(loc='upper right')
plt.show()
