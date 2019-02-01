import numpy as np 
import pandas as pd 
import random
import skimage
import keras

from scipy.misc import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D

train_data = pd.read_csv('../input/train/train.csv')
test_data = pd.read_csv('../input/test/test.csv')

def Preprocessing(path, data):
    temp_array = []

    for image_ID in data.ID:
        image = plt.imread(path + str(image_ID))
        image = skimage.transform.resize(image, (24, 24))
        temp_array.append(image.astype('float32'))
    
    img = np.stack(temp_array)
    img = img / 255
    return img

def Convolutional_Neural_Network():
    model = Sequential()

    model.add(Conv2D(40, kernel_size=5, padding="same", input_shape=(24, 24, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(50, kernel_size=5, padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(70, kernel_size=3, padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(100, kernel_size=3, padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(output_dim=500, activation='relu'))
    model.add(Dense(output_dim=3, input_dim=500))
    model.add(Activation('softmax'))
    
    return model

def index():
    i = random.choice(train_data.index)
    return i

def print_img():
    i = index()
    image_ID = train_data.ID[i]
    image = plt.imread('../input/train/Train/' + str(image_ID))
    image.astype('float32')
    image = skimage.transform.resize(image, (150, 150))
    plt.imshow(image)

def visualize_results():
    i = index()
    print_img()
    prediction = model.predict_classes(X_train)
    print('Actual:', train_data.Class[i], 'Predicted:', 
          le.inverse_transform(prediction[i]))

X_train = Preprocessing('../input/train/Train/', train_data)
X_test = Preprocessing('../input/test/Test/', test_data)

le = LabelEncoder()
Y_train = le.fit_transform(train_data.Class)
Y_train = keras.utils.np_utils.to_categorical(Y_train)

model = Convolutional_Neural_Network()
model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=250, epochs=100, 
          validation_split = 0.2)

visualize_results()
