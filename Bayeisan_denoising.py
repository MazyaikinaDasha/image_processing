from degrading_image_tools import DegradationType, DegradingImage
import imageio.v2 as imageio
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
degraded_img = degrading_image.degradation(img)

reconst = bayes_denoise(degraded_img, iters = 7, surplus = True)
imageio.imwrite('Image_from_array.png', reconst[-1])
reconst_img = imageio.imread('Image_from_array.png')

print(compute_psnr(img, reconst_img))

