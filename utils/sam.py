"""
Utility functions for Meta's Segment Anything Model (SAM)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def show_mask(mask, ax, random_color=False):
    """
    Show a mask on a given axis

    Parameters
    ----------
    mask : torch.Tensor
        Mask to show
    ax : matplotlib.axes.Axes
        Axis to show the mask on
    random_color : bool, optional
        Whether to use a random color for the mask, by default False
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """
    Show point prompts on a given axis

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates of the points
    labels : torch.Tensor
        Labels of the points
    ax : matplotlib.axes.Axes
        Axis to show the points on
    marker_size : int, optional
        Size of the markers, by default 375
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    """
    Show a bounding box (prompt) on a given axis

    Parameters
    ----------
    box : torch.Tensor
        Bounding box to show
    ax : matplotlib.axes.Axes
        Axis to show the box on
    """

    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
