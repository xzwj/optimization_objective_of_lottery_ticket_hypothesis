import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

import utils
import models.nets as nets


def get_mean_of_all_weights(pruner_by_epoch):
    """
    Get the mean of all weights for each pruner in the list of pruners.
    Args:
        pruner_by_epoch: list of pruners.
    :return: a list of mean values.
    """
    mean_by_epoch = []
    for pruner in pruner_by_epoch:
        flat_masks = pruner.get_flat_masks().detach().numpy()
        mean_by_epoch.append(np.mean(flat_masks))
    return mean_by_epoch


def get_mean_of_remaining_weights(pruner_by_epoch):
    threshold = 0.001
    mean_by_epoch = []
    for pruner in pruner_by_epoch:
        flat_masks = pruner.get_flat_masks().detach().numpy()
        mean_by_epoch.append(np.mean(flat_masks[flat_masks > threshold]))
    return mean_by_epoch