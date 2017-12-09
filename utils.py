import numpy as np
import tensorflow as tf
from scipy.misc import imsave as ims

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image
        ims("individual_images/" + str(idx)+".jpg", image)

    return img
