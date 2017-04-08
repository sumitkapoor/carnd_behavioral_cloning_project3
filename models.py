from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, SpatialDropout2D, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

class Models(object):

    def __init__(self, mtype='nvidia', batch_size=32, input_shape=(160, 320, 3)):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.model = None

        if mtype == 'nvidia':
            self.model = self._create_nvidia_model()

    def _create_nvidia_model(self):
        model = Sequential()
        model.add(Lambda(lambda x: x, input_shape=self.input_shape))
        model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2), padding="same"))
        model.add(SpatialDropout2D(0.2))
        model.add(Conv2D(36, (5, 5), padding="same", strides=(2, 2), activation="elu"))
        model.add(SpatialDropout2D(0.2))
        model.add(Conv2D(48, (5, 5), padding="valid", strides=(2, 2), activation="elu"))
        model.add(SpatialDropout2D(0.2))
        model.add(Conv2D(64, (3, 3), padding="valid", activation="elu"))
        model.add(SpatialDropout2D(0.2))
        model.add(Conv2D(64, (3, 3), padding="valid", activation="elu"))
        model.add(SpatialDropout2D(0.2))
        model.add(Flatten())

        model.add(Dense(1164, activation="elu"))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation="elu"))
        model.add(Dense(50, activation="elu"))
        model.add(Dense(10, activation="elu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="linear"))

        return model
