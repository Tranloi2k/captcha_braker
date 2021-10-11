from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras_preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from Convolution_Network import CN
from captchahelper import preprocess
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

def step_decay(epoch):

    # initialize the base initial learning rate, drop factor, and epochs to drop every
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 10

    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    # return the learning rate
    return float(alpha)

callbacks = [LearningRateScheduler(step_decay)]

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.01, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


# initialize data and label
data = []
labels = []

for imagePath in paths.list_images(args["dataset"]):

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 40, 40)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

data = np.array(data, dtype=float) / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer().fit(trainY)
trainY = lb.transform(trainY)
testY = lb.transform(testY)

# initialize the model
print("[INFO] compiling model...")
model = CN.build(width=40, height=40, depth=1, classes=9)
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
epoch= 50
BS = 128
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // BS,
              epochs=epoch, callbacks=callbacks, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch), H.history["accuracy"], label="accuracy")
plt.plot(np.arange(0, epoch), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


#python train_model.py --dataset C:\Annotate --model C:\Users\Admin\PycharmProjects\brakecapcha\lenet.hdf5

#python test_model.py --input C:\Datasets  --model C:\Users\Admin\PycharmProjects\brakecapcha\lenet.hdf5
