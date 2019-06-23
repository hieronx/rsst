import numpy as np
import cv2

from .augmentations import SSDAugmentation

def extract_image_patch(image, bbox):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """

    # x, y, w, h = bbox.astype(np.int)
    # return image[y:y+h, x:x+w]

    bbox = np.array(bbox)

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    return image

