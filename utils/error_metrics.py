"""
In order to facilitate quick comparison of different OSSEs, I need to quickly
be able to compare the error metrics of different OSSEs.

Author:        Mike Stanley
Created:       20 Dec 2019
Last Modified: 20 Dec 2019

===============================================================================
"""

import numpy as np


def eliminate_oceans(arr1, arr2):
    """
    Find boolean mask of fluxes that eliminates oceans

    Parameters:
        arr1 (numpy arr) : 2d array
        arr2 (numpy arr) : 2d array

    Returns:
        2d numpy arr where coordinate is FALSE where both are 0.
        This creates a boolean mask for both arrays such that we
        eliminate all coordinates where array 1 is

    NOTE:
    - We use the logical AND because we want to retain the two important
      cases:
          1. true is 0 but flux is not
          2. true is non zero but flux is 0

      The only points it appears that we for certain want to eliminate are
      those where both arrays are 0.
    """
    # determine where not 0
    arr1_0 = arr1 == 0
    arr2_0 = arr2 == 0

    return ~(arr1_0 & arr2_0)


def compute_mse(true_arr, pred_arr):
    """
    Find MSE between two arrays after eliminating oceans

    Parameters:
        true_arr (numpy arr) : 2d/3d array of true fluxes
        pred_arr (numpy arr) : 2d/3d array of predicted fluxes (prior or
                               posterior)

    Returns:
        float of MSE between arrays

    NOTE:
    - can for for both monthly fluxes (2d) and 3d fluxes (over time and space)
    """
    # find land
    land_mask = eliminate_oceans(true_arr, pred_arr)
    num_land_coord = land_mask.sum()

    return np.square(true_arr[land_mask] - pred_arr[land_mask]).sum() / num_land_coord


def compute_mad(true_arr, pred_arr):
    """
    Find the median absolute difference after eliminating oceans

    Parameters:
        true_arr (numpy arr) : 2d/3d array of true fluxes
        pred_arr (numpy arr) : 2d/3d array of predicted fluxes (prior or
                               posterior)

    Returns:
        float of MAD between arrays
    """
    # find land
    land_mask = eliminate_oceans(true_arr, pred_arr)

    return np.median(np.abs(true_arr[land_mask] - pred_arr[land_mask]))


def compute_rmse(true_arr, pred_arr):
    """
    Find the RMSE after eliminating oceans
    """
    # find the mse
    mse = compute_mse(true_arr, pred_arr)

    return np.sqrt(mse)


def compute_mse_months(true_arr, pred_arr):
    """
    Computes monthly MSE

    Parameters:
        true_arr (numpy arr) : 3d array of true fluxes
        pred_arr (numpy arr) : 3d array of predicted fluxes (prior or
                               posterior)

    Returns:
        list of monthly MSEs for each coordinate in first dimension of arrays

    NOTE:
    - we enforce that the given arrays are 3d
    """
    assert len(true_arr.shape) == 3
    assert len(pred_arr.shape) == 3
    assert true_arr.shape[0] == pred_arr.shape[0]

    monthly_mses = []

    for i in range(true_arr.shape[0]):
        monthly_mses.append(
            compute_mse(true_arr[i, :, :], pred_arr[i, :, :])
        )

    return monthly_mses


def compute_rmse_months(true_arr, pred_arr):
    """
    Computes monthly RMSE

    Parameters:
        true_arr (numpy arr) : 3d array of true fluxes
        pred_arr (numpy arr) : 3d array of predicted fluxes (prior or
                               posterior)

    Returns:
        list of monthly RMSEs for each coordinate in first dimension of arrays

    NOTE:
    - we enforce that the given arrays are 3d
    """
    # compute monthly MSEs
    mses = compute_mse_months(true_arr, pred_arr)

    return [np.sqrt(mse) for mse in mses]


def compute_mad_months(true_arr, pred_arr):
    """
    Compute monthly median absolute difference

    Parameters:
        true_arr (numpy arr) : 3d array of true fluxes
        pred_arr (numpy arr) : 3d array of predicted fluxes (prior or
                               posterior)

    Returns:
        list of monthly MADs for each coordinate in first dimension of arrays

    NOTE:
    - we enforce that the given arrays are 3d
    """
    assert len(true_arr.shape) == 3
    assert len(pred_arr.shape) == 3
    assert true_arr.shape[0] == pred_arr.shape[0]

    monthly_mads = []

    for i in range(true_arr.shape[0]):
        monthly_mads.append(
            compute_mad(true_arr[i, :, :], pred_arr[i, :, :])
        )

    return monthly_mads
