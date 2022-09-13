import numpy as np
import math
from scipy.constants.constants import pi
from numpy.ma.core import exp
import scipy.ndimage

def compute_ssim(y_original, y_estimated):
    # COMPUTE_SSIM Computes the SSIM between two images
    #
    # Input:
    #  y_original  - The original image
    #  y_estimated - The estimated image
    #
    # Output:
    #  ssim_val - Structural similarity index
    # Variables for Gaussian kernel definition
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))

    # Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i, j] = \
                (1 / (2 * pi * (gaussian_kernel_sigma ** 2))) * \
                exp(-(((i - 5) ** 2) + ((j - 5) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))

    # Convert image matrices to double precision (like in the Matlab version)
    img_mat_1 = y_original.astype(np.float)
    img_mat_2 = y_estimated.astype(np.float)

    # Squares of input matrices
    img_mat_1_sq = img_mat_1 ** 2
    img_mat_2_sq = img_mat_2 ** 2
    img_mat_12 = img_mat_1 * img_mat_2

    # Means obtained by Gaussian filtering of inputs
    img_mat_mu_1 = scipy.ndimage.filters.convolve(img_mat_1, gaussian_kernel)
    img_mat_mu_2 = scipy.ndimage.filters.convolve(img_mat_2, gaussian_kernel)

    # Squares of means
    img_mat_mu_1_sq = img_mat_mu_1 ** 2
    img_mat_mu_2_sq = img_mat_mu_2 ** 2
    img_mat_mu_12 = img_mat_mu_1 * img_mat_mu_2

    # Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq = scipy.ndimage.filters.convolve(img_mat_1_sq, gaussian_kernel)
    img_mat_sigma_2_sq = scipy.ndimage.filters.convolve(img_mat_2_sq, gaussian_kernel)

    # Covariance
    img_mat_sigma_12 = scipy.ndimage.filters.convolve(img_mat_12, gaussian_kernel)

    # Centered squares of variances
    img_mat_sigma_1_sq = img_mat_sigma_1_sq - img_mat_mu_1_sq
    img_mat_sigma_2_sq = img_mat_sigma_2_sq - img_mat_mu_2_sq
    img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12;

    # c1/c2 constants
    # First use: manual fitting
    c_1 = 6.5025
    c_2 = 58.5225

    # Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l = 255
    k_1 = 0.01
    c_1 = (k_1 * l) ** 2
    k_2 = 0.03
    c_2 = (k_2 * l) ** 2

    # Numerator of SSIM
    num_ssim = (2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2)
    # Denominator of SSIM
    den_ssim = (img_mat_mu_1_sq + img_mat_mu_2_sq + c_1) * \
               (img_mat_sigma_1_sq + img_mat_sigma_2_sq + c_2)
    # SSIM
    ssim_map = num_ssim / den_ssim
    index = np.average(ssim_map)
    return index

def compute_psnr(y_original, y_estimated):
    # COMPUTE_PSNR Computes the PSNR between two images
    #
    # Input:
    #  y_original  - The original image
    #  y_estimated - The estimated image
    #
    # Output:
    #  psnr_val - The Peak Signal to Noise Ratio (PSNR) score

    y_original = np.reshape(y_original, (-1))
    y_estimated = np.reshape(y_estimated, (-1))

    # Compute the dynamic range
    # Write your code here... dynamic_range = ????
    dynamic_range = 255.0

    # Compute the Mean Squared Error (MSE)
    # Write your code here... mse_val = ????
    mse_val = (1 / len(y_original)) * np.sum((y_original - y_estimated) ** 2)

    # Compute the PSNR
    # Write your code here... psnr_val = ????
    psnr_val = 10 * math.log10(dynamic_range ** 2 / mse_val)

    return psnr_val


def compute_stat(est_patches, orig_patches, est_coeffs):
    # COMPUTE_STAT Compute and print usefull statistics of the pursuit and
    # learning procedures
    #
    # Inputs:
    #  est_patches  - A matrix, containing the recovered patches as its columns
    #  orig_patches - A matrix, containing the original patches as its columns
    #  est_coeffs   - A matrix, containing the estimated representations,
    #                 leading to est_patches, as its columns
    #
    # Outputs:
    #  residual_error  - Average Mean Squared Error (MSE) per pixel
    #  avg_cardinality - Average number of nonzeros that is used to represent
    #                    each patch
    #

    # Compute the Mean Square Error per patch
    MSE_per_patch = np.sum((est_patches - orig_patches) ** 2, axis=0)

    # Compute the average
    residual_error = np.mean(MSE_per_patch) / np.shape(orig_patches)[0]

    # Compute the average number of non-zeros
    avg_cardinality = np.sum(np.abs(est_coeffs) > 10 ** (-10)) / np.shape(est_coeffs)[1]

    # Display the results
    print('Residual error %2.2f, Average cardinality %2.2f' % (residual_error, avg_cardinality))

    return residual_error, avg_cardinality
