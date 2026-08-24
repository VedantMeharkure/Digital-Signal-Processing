import cv2
import numpy as np

NAME = "Histogram Equalization"

PARAMS = {
    "method": {
        "type": "select",
        "label": "Method",
        "options": [
            "Global",
            "Local",
            "Adaptive"
        ],
        "default": "Global"
    },

    "window_size": {
        "type": "slider",
        "label": "Local Window Size",
        "min": 3,
        "max": 101,
        "step": 2,
        "default": 51
    },

    "clip_limit": {
        "type": "slider",
        "label": "CLAHE Clip Limit",
        "min": 1.0,
        "max": 40.0,
        "step": 0.5,
        "default": 2.0
    },

    "tile_size": {
        "type": "slider",
        "label": "CLAHE Tile Grid Size",
        "min": 2,
        "max": 16,
        "step": 1,
        "default": 8
    }
}


def apply(image, params):

    method = params["method"]

    # Convert RGB → YCrCb
    ycrcb = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2YCrCb
    )

    y = ycrcb[:, :, 0]

    # -----------------------------
    # GLOBAL
    # -----------------------------

    if method == "Global":

        y_equalized = cv2.equalizeHist(y)

    # -----------------------------
    # ADAPTIVE
    # -----------------------------

    elif method == "Adaptive":

        clip_limit = params["clip_limit"]
        tile_size = params["tile_size"]

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )

        y_equalized = clahe.apply(y)

    # -----------------------------
    # LOCAL
    # -----------------------------

    elif method == "Local":

        window = params["window_size"]

        y_float = y.astype(np.float32)

        mean = cv2.boxFilter(
            y_float,
            -1,
            (window, window),
            normalize=True
        )

        mean_sq = cv2.boxFilter(
            y_float * y_float,
            -1,
            (window, window),
            normalize=True
        )

        variance = mean_sq - mean * mean

        std = np.sqrt(
            np.maximum(variance, 0)
        )

        # Local contrast enhancement
        result = (
            (y_float - mean)
            * (64.0 / (std + 1.0))
            + mean
        )

        y_equalized = np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

    else:

        raise ValueError(
            "Invalid histogram equalization method"
        )

    # Put processed Y back
    ycrcb[:, :, 0] = y_equalized

    result = cv2.cvtColor(
        ycrcb,
        cv2.COLOR_YCrCb2RGB
    )

    return result