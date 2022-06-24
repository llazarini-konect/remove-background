
from keras.models import load_model
import cv2
import numpy as np
import imageio
from PIL import Image

from glob import glob
from tensorflow.python.client import device_lib;print(device_lib.list_local_devices())
from keras import backend as K
from keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.layers import BatchNormalization

IMG_ROWS = 480
IMG_COLS = 320

SMOOTH = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


model = load_model('model.keras', custom_objects={"bce_dice_loss": bce_dice_loss, "dice_coef": dice_coef});

#image = sorted(glob('./test/*.jpg'))[:32]
imgSrc = './test/0a2bbd5330a2_10.jpg'
image = glob(imgSrc)
images = np.array([cv2.resize(imageio.imread(path), (IMG_ROWS, IMG_COLS))
    for path in image])
                        
# Predict image
bnorm1 = BatchNormalization()(images)
prediction = model.predict(bnorm1)
        
print(np.squeeze(prediction))
img = Image.fromarray(np.squeeze(prediction))

img = cv2.imread(imgSrc)
img = cv2.resize(img, (IMG_ROWS, IMG_COLS))

mask = np.squeeze(prediction)

img[mask] = 0

print(img)

mask = cv2.imwrite('teste5.png', img)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)



#cv2.fillPoly(mask, img, 255)

cv2.imwrite('mask1.png', mask)
cv2.imwrite('mask2.png', img)

#es = cv2.bitwise_and(img, img, mask = mask)

