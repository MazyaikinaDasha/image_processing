from degrading_image_tools import DegradationType, DegradingImage
import imageio.v2 as imageio
import numpy as np
import cv2
from denoise import bayes_denoise
from compute_stat import *



image_data_path = "sample_data/barbara.png"
config_image_degradation = {
    "image_degradation_type": DegradationType.add_random_noise,
    "noise_sigma": 10, "random_seed": 1
}# Load an image
img = imageio.imread(image_data_path)
img = img[:256, 251:251+256]



# Degrading an image
degrading_image = DegradingImage(sigma=config_image_degradation["noise_sigma"],
                                 rand_seed=config_image_degradation["random_seed"],
                                 degradation_type=config_image_degradation["image_degradation_type"])
im = degrading_image.degradation(img)

from scipy import fftpack

im_fft = fftpack.fft2(im)
keep_fraction = 0.1

# Call ff a copy of the original transform. Numpy arrays have a copy
# method for this purpose.
im_fft2 = im_fft.copy()

# Set r and c to be the number of rows and columns of the array.
r, c = im_fft2.shape

# Set to zero all rows with indices between r*keep_fraction and
# r*(1-keep_fraction):
im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0

# Similarly with the columns:
im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
im_new = fftpack.ifft2(im_fft2).real
print(compute_psnr(img, im_new))