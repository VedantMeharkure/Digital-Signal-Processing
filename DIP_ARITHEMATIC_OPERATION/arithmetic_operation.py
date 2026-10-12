
import cv2
import numpy as np

NAME = "Arithmetic Operations"

PARAMS = {
    "operation": {
        "type": "select",
        "label": "Select Arithmetic Operation",
        "options": [
            "Addition",
            "Subtraction",
            "Multiplication",
            "Division"
        ],
        "default": "Addition"
    },
    "constant": {
        "type": "slider",
        "label": "Constant Value",
        "min": 1,
        "max": 255,
        "default": 50,
        "step": 1
    }
}


def apply(image, params):
    operation = params.get("operation", "Addition")
    constant = params.get("constant", 50)

    # Convert input to NumPy array
    img = np.asarray(image)

    # Use float for calculations to avoid uint8 overflow
    img_float = img.astype(np.float32)

    if operation == "Addition":
        result = img_float + constant

    elif operation == "Subtraction":
        result = img_float - constant

    elif operation == "Multiplication":
        result = img_float * (constant / 50.0)

    elif operation == "Division":
        result = img_float / (constant / 50.0)

    else:
        raise ValueError("Invalid arithmetic operation")

    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)
