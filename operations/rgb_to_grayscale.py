import cv2
import numpy as np

NAME = "RGB to Grayscale"

PARAMS = {
    "method": {
        "type": "select",
        "label": "Method",
        "options": [
            "Luminosity",
            "Average",
            "Lightness"
        ],
        "default": "Luminosity"
    }
}


def apply(image, params):

    method = params["method"]

    # RGB image coming from Streamlit
    if method == "Luminosity":

        gray = (
            0.299 * image[:, :, 0] +
            0.587 * image[:, :, 1] +
            0.114 * image[:, :, 2]
        )

    elif method == "Average":

        gray = np.mean(image, axis=2)

    elif method == "Lightness":

        gray = (
            np.max(image, axis=2) +
            np.min(image, axis=2)
        ) / 2

    else:
        raise ValueError("Unknown grayscale method")

    return gray.astype(np.uint8)