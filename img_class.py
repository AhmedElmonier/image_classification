import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#print(type(x_train))
#print(type(y_train))
#print(type(x_test))
#print(type(y_test))

#print('X Train shape: ', x_train.shape)
#print('Y Train shape: ', y_train.shape)
#print('X Test shape: ', x_test.shape)
#print('y Test shape: ', y_test.shape)

index = 500
#img = x_train[index]
#print('This image label is: ', classification[y_train[index][0]])
#plt.imshow(img)
#plt.show()

# Convert the labels to a set of 10 numbers
y_train_one_hot = to_categorical(y_train) 
y_test_one_hot = to_categorical(y_test)

#print(y_train_one_hot)
#print('The one hot label is: ', y_train_one_hot[index])

x_train = x_train / 255
x_test = x_test / 255

# Creating the model
model = Sequential()
# adding first layer
model.add(Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)))
#adding a pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
# adding second layer
model.add(Conv2D(32, (5,5), activation='relu'))
# adding another pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
# adding Flatten layer
model.add(Flatten())
# Adding a layer with 1000 neurons
model.add(Dense(1000, activation='relu'))
#add dropout layer
model.add(Dropout(0.5))
# Adding a layer with 500 neurons
model.add(Dense(500, activation='relu'))
#add dropout layer
model.add(Dropout(0.5))
# Adding a layer with 250 neurons
model.add(Dense(250, activation='relu'))
# Adding a layer with 10 neurons
model.add(Dense(10, activation='softmax'))

# Compiling th model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
hist = model.fit(x_train, y_train_one_hot,
                 batch_size = 256,
                 epochs = 10,
                 validation_split=0.2)

# evaluate the model 
model.evaluate(x_test, y_test_one_hot)[1]

# Visualize the Accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'])
plt.savefig('Accuracy_vis.pdf')
#plt.show()

# Visualize the Loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'])
plt.savefig('Loss_vis.pdf')
#plt.show()
