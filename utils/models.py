from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, SpatialDropout2D, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

class Models(object):

    def __init__(self, mtype='nvidia', batch_size=32, input_shape=(160, 320, 3)):
        """
        INPUT:
            mtype : The type of the model that needs to be created, default nvidea
            batch_size : Batch size for model training
            input_shape : input shape of the data that the model will use for training

        Creates a model based on the model type. Model can be refrenced using public attribute model.

        """
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.model = None

        if mtype == 'nvidia':
            self.model = self._create_nvidia_model()

    def _create_nvidia_model(self):
        """
        Based on the architecture published by Nvidia:
        http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

        The model starts with the Lamba layer where the image is normalized.

        The model then has 5 convolution layers, followed by a dropout layer and then 3 Dense layer.
        The output for the model is 1 unit.

        ELU has been used as activation unit.

        l2 regularization has also been used at each layer.
        """
        # Using l2 regularization
        model = Sequential()

        model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape = self.input_shape))
        model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="same", W_regularizer = l2(0.001), activation="elu"))
        model.add(Conv2D(36, 5, 5, border_mode="same", subsample=(2, 2), W_regularizer = l2(0.001), activation="elu"))
        model.add(Conv2D(48, 5, 5, border_mode="valid", subsample=(2, 2), W_regularizer = l2(0.001), activation="elu"))
        model.add(Conv2D(64, 3, 3, border_mode="valid", W_regularizer = l2(0.001), activation="elu"))
        model.add(Conv2D(64, 3, 3, border_mode="valid", W_regularizer = l2(0.001), activation="elu"))

        model.add(Flatten())

        # added dropout
        model.add(Dropout(0.5))

        # Removed Dense(1164), increased unitst from 100, 50, 10
        model.add(Dense(256, activation="elu", W_regularizer = l2(0.001)))
        model.add(Dense(96, activation="elu", W_regularizer = l2(0.001)))
        model.add(Dense(24, activation="elu", W_regularizer = l2(0.001)))

        model.add(Dense(1))

        return model
