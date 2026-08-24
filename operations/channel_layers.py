import numpy as np

NAME = "RGB Channel Layers"

PARAMS = {
    "channel": {
        "type": "select",
        "label": "Channel",
        "options": [
            "Red",
            "Green",
            "Blue",
            "All"
        ],
        "default": "All"
    },

    "display": {
        "type": "select",
        "label": "Display",
        "options": [
            "Grayscale",
            "Color"
        ],
        "default": "Grayscale"
    }
}


def apply(image, params):

    channel = params["channel"]
    display = params["display"]

    if channel == "Red":
        index = 0

    elif channel == "Green":
        index = 1

    elif channel == "Blue":
        index = 2

    else:

        return image

    selected = image[:, :, index]

    if display == "Grayscale":
        return selected

    result = np.zeros_like(image)
    result[:, :, index] = selected

    return result