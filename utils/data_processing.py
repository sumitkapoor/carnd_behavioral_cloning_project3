import cv2
import numpy as np

def brighten_augmentation(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype = np.float64)
    random_bright = .5 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255]  = 255
    img = np.array(img, dtype = np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img

def trans_image(image, steer, trans_x_range, trans_y_range):
    rows, cols = image.shape[:-1]
    tr_x = trans_x_range * np.random.uniform() - trans_x_range/ 2
    steer_ang = steer + tr_x/trans_x_range * 2 * 0.2
    tr_y = trans_y_range * np.random.uniform() - trans_y_range/ 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))
    
    return image_tr, steer_ang

def flip_image(image, steer):
    flipped_image = cv2.flip(image, 1)
    flipped_steer = -1 * steer
    return flipped_image, flipped_steer

def preprocess_image(image):
    image = image[60:140, 0:320]
    image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def generate_image(image, angle):
    image = preprocess_image(image)
    image, angle = trans_image(image, angle, 50, 10)
    image = brighten_augmentation(image)
    return image, angle
