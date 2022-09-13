import numpy as np
from enum import Enum

class DegradationType(Enum):
    add_random_noise = 0
    missing_pixels = 1

class DegradingImage:
    def __init__(self, sigma=20, rand_seed=1,
                 degradation_type=DegradationType.add_random_noise):

        self.sigma = sigma
        self.rand_seed = rand_seed
        self.degradation_type = degradation_type

        np.random.seed(self.rand_seed)


    def degradation(self, img):
        if self.degradation_type == DegradationType.add_random_noise:
            K = (np.random.rand(img.shape[0], img.shape[1]) < .99).astype(int)
            #cor_img = np.ones(img.shape)
            #cor_img[K == 1] = img[K == 1]
            #noisy_img = img + np.random.normal(loc=0, scale=self.sigma, size=img.shape)
            noisy_img = self.salt_and_pepper(img)
            return noisy_img

    def salt_and_pepper (self, img):
        row, col = img.shape
        s_vs_p = 0.5
        amount = 0.009
        out = img
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        out[coords] = 0
        return out


