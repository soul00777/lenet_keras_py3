import numpy as np
import sys
import cv2
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.datasets import mnist
from keras.utils import np_utils


# for how to plot the digit, see https://stackoverflow.com/questions/37228371/visualize-mnist-dataset-using-opencv-or-matplotlib-pyplot
# credit https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
# credit https://elitedatascience.com/keras-tutorial-deep-learning-in-python
# credit https://keras.io/


(data_train, label_train), (data_test, label_test) = mnist.load_data()
# print("Data_train shape is ",data_train.shape)
data_train = data_train.reshape(data_train.shape[0], 28, 28, 1)
# print("Data_train shape is ",data_train.shape)
print_data_test = data_test
data_test = data_test.reshape(data_test.shape[0], 28, 28, 1)
data_train = data_train.astype("float32")
data_test = data_test.astype("float32")
data_train /= 255
data_test /= 255
# print("Label_train shape is ", label_train.shape)
label_train = np_utils.to_categorical(label_train, 10)
label_test = np_utils.to_categorical(label_test, 10)
# print("Label_train shape is ",label_train.shape)

model = Sequential()
model.add(Conv2D(20, kernel_size=(5, 5), strides=(1, 1), padding="valid", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"))
model.add(Activation("tanh"))
model.add(SeparableConv2D(50, kernel_size=(5, 5), strides=(1, 1), padding="valid", depth_multiplier=20))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"))
model.add(Activation("tanh"))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("tanh"))
model.add(Dense(10))
model.add(Activation("softmax"))


print("[INFO] compiling model...")
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

train = input("Do you want to train the LeNet now? (Y for train, N for NOT to train and load the pre-trained weight from hdf5 file) ")
if train == "Y":
    print("[INFO] training...")
    model.fit(data_train, label_train, batch_size=400, epochs=1, verbose=1)
    wname = input("Enter the name of weights file you want to save: ")
    print("[INFO] overwrite weights to file...")
    model.save_weights(wname, overwrite=True)
    print("[INFO] file saved...")
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

for i in np.random.choice(np.arange(0, len(label_test)), size=(10,)):
    probs = model.predict(data_test[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    image = print_data_test[i]
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, str(prediction[0]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], np.argmax(label_test[i])))
    cv2.imshow("Digit", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
