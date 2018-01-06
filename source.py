import numpy as np
import sys
from keras.models import Sequential
from keras import optimizers
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense, Flatten
from keras.datasets import mnist
from keras.utils import np_utils
from keras import initializers

(data_train, label_train), (data_test, label_test) = mnist.load_data()
data_train = data_train.reshape(data_train.shape[0], 28, 28, 1)
data_test = data_test.reshape(data_test.shape[0], 28, 28, 1)
data_train = data_train.astype("float32")
data_test = data_test.astype("float32")
data_train /= 255
data_test /= 255
label_train = np_utils.to_categorical(label_train, 10)
label_test = np_utils.to_categorical(label_test, 10)

model = Sequential()
model.add(Conv2D(20, kernel_size=(5, 5), strides=(1, 1), padding="valid", input_shape=(28, 28, 1), kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(Conv2D(50, kernel_size=(5, 5), strides=(1, 1), padding="valid", kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(500,kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(10,kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
model.add(Activation("softmax"))

print("[INFO] compiling model...")
model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(momentum=0.9, nesterov=True), metrics=["accuracy"])

train = input("Do you want to train the LeNet now? (Y for train, N for NOT to train and load the pre-trained weight from hdf5 file) ")
if train == "Y":
    print("[INFO] training...")
    model.fit(data_train, label_train, batch_size=128, epochs=200, verbose=1)
    wname = input("Enter the name of weights file you want to save: ")
    model_json = model.to_json()
    with open("lenet5.json", "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    print("[INFO] model saved as lenet5.json...")
    model.save_weights(wname, overwrite=True)
    print("[INFO] weights saved...")
elif train == "N":
    wname = input("Enter the weight file you want to load: ")
    print("[INFO] loading weight...")
    model.load_weights(wname)
    print("[INFO] weight loaded...")
else:
    print("[ERROR] Invalid input (now exit the program)...")
    sys.exit(0)

print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(data_test, label_test, batch_size=10000, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
print("[INFO] loss: {:.2f}%".format(loss * 100))
