from keras.applications.vgg16 import VGG16
from keras.datasets import cifar100
from keras.models import Model, load_model, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers


num_classes=10
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()
print(train_X.shape)
print(test_X.shape)
img_height, img_width, channel = train_X.shape[1],train_X.shape[2],1

train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1,28,28,1)

# normalize data
train_X = train_X.astype("float32")
test_X = test_X.astype("float32")
train_X = train_X/255
test_X = test_X/255

test_Y_one_hot = to_categorical(test_Y, num_classes=num_classes)
train_Y_one_hot = to_categorical(train_Y, num_classes=num_classes)
# (3) Create a sequential model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=48, input_shape=(img_height, img_width, channel,), kernel_size=(4,4),\
 strides=(4,4), padding='same'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(28*28*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(512))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(256))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

train_x_vgg = model.predict(train_X)
test_x_vgg = model.predict(test_X)

import tensorflow.keras 

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors =1) 
knn.fit(train_x_vgg,train_Y_one_hot)
pred=knn.predict(test_x_vgg)

print("Acc: ",accuracy_score(test_Y_one_hot, pred))