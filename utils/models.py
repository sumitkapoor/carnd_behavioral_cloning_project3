from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, SpatialDropout2D, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

class Models(object):

    def __init__(self, mtype='nvidia', batch_size=32, input_shape=(160, 320, 3)):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.model = None

        if mtype == 'nvidia':
            self.model = self._create_nvidia_model()

    def _create_nvidia_model(self):
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape = self.input_shape))
        model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="same", W_regularizer = l2(0.001), activation="elu"))
        #model.add(SpatialDropout2D(0.7))
        model.add(Conv2D(36, 5, 5, border_mode="same", subsample=(2, 2), W_regularizer = l2(0.001), activation="elu"))
        #model.add(SpatialDropout2D(0.7))
        model.add(Conv2D(48, 5, 5, border_mode="valid", subsample=(2, 2), W_regularizer = l2(0.001), activation="elu"))
        #model.add(SpatialDropout2D(0.7))
        model.add(Conv2D(64, 3, 3, border_mode="valid", W_regularizer = l2(0.001), activation="elu"))
        #model.add(SpatialDropout2D(0.7))
        model.add(Conv2D(64, 3, 3, border_mode="valid", W_regularizer = l2(0.001), activation="elu"))


        model.add(Flatten())

        model.add(Dropout(0.5))

        model.add(Dense(256, activation="elu", W_regularizer = l2(0.001)))
        model.add(Dense(96, activation="elu", W_regularizer = l2(0.001)))
        model.add(Dense(24, activation="elu", W_regularizer = l2(0.001)))
        model.add(Dense(1))

        return model
