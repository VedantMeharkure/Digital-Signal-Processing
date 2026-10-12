import cv2
import numpy as np
import pywt
NAME = "Haar Wavelet Transform"
PARAMS = {
    "level": {
        "type": "slider",
        "label": "Decomposition Level",
        "min": 1,
        "max": 3,
        "step": 1,
        "default": 1
    },
    "component": {
        "type": "select",
        "label": "Component",
        "options": [
            "Approximation",
            "Horizontal Detail",
            "Vertical Detail",
            "Diagonal Detail"
        ],
        "default": "Approximation"
    }
}
def apply(image, params):
    level = params["level"]
    component = params["component"]
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )
    coefficients = pywt.wavedec2(
        gray.astype(float),
        "haar",
        level=level
    )
    approximation = coefficients[0]
    details = coefficients[1:]
    if component == "Approximation":
        result = approximation
    else:

        horizontal, vertical, diagonal = (
            details[0]
        )
        if component == "Horizontal Detail":
            result = horizontal
        elif component == "Vertical Detail":
            result = vertical
        else:
            result = diagonal

    # Normalize coefficient values
    result = cv2.normalize(
        result,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    result = result.astype(
        np.uint8
    )

    return result