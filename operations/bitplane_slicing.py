import cv2
import numpy as np

NAME = "Bit-plane Slicing"

PARAMS = {
    "bit": {
        "type": "slider",
        "label": "Bit Plane",
        "min": 0,
        "max": 7,
        "step": 1,
        "default": 7
    },

    "mode": {
        "type": "select",
        "label": "Mode",
        "options": [
            "Single Plane",
            "All Planes"
        ],
        "default": "Single Plane"
    }
}


def apply(image, params):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    mode = params["mode"]

    if mode == "Single Plane":

        bit = params["bit"]

        result = ((gray >> bit) & 1) * 255

        return result.astype(np.uint8)

    # All 8 bit planes
    planes = []

    for bit in range(8):

        plane = ((gray >> bit) & 1) * 255
        planes.append(plane.astype(np.uint8))

    return planes