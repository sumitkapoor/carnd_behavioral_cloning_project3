import numpy as np
import random
import sklearn

from utils.data_processing import *

def behaviour_cloning_generator(data, batch_size = 128):
    """"
    INPUT:
         data: list of tuples of the form:
         (left_image_path, center_image_path, right_image_path, steer_angle)
    OUTPUT:
         returns a generator with X_train as a numpy array of image and y_train
         as a numpy array of corresponding steering angles
    """
    num_samples = len(data)
    while 1:
        np.random.shuffle(data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                select_camera = np.random.randint(3)
                #
                # Randomly select left, right and center camera
                #
                img_path = batch_sample[select_camera]

                shift = 0
                if select_camera != 1:
                    # if camera angle is left, adjust angle by +0.2
                    # and -0.2 if it's right 
                    shift = 0.2 if select_camera < 1 else -0.2

                y_angle = batch_sample[3]

                if abs(y_angle) > 0.15:
                    y_angle += shift

                image = cv2.imread(img_path)

                generate_or_not = np.random.randint(2)

                if not generate_or_not:
                    x_image = preprocess_image(image)
                else:
                    # 
                    # Randomly flip the image or just brighten
                    # and tranmutate
                    # 
                    x_image, y_angle = generate_image(image, y_angle)

                # flip 50% of the time
                if random.random() < 0.5:
                    x_image, y_angle = flip_image(x_image, y_angle)

                images.append(x_image)
                angles.append(y_angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            yield (X_train, y_train)
