
import cv2
import numpy as np

NAME = "Edge Detection"

PARAMS = {
    "method": {
        "type": "select",
        "label": "Edge Detection Method",
        "options": ["Sobel", "Prewitt", "Roberts", "Laplacian", "Canny"],
        "default": "Canny"
    },
    "threshold": {
        "type": "slider",
        "label": "Edge Threshold",
        "min": 0,
        "max": 255,
        "default": 30,
        "step": 5
    }
}


def apply(image, params):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)

    method = params.get("method", "Canny")
    threshold = params.get("threshold", 30)

    if method == "Sobel":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        edges = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        edges = cv2.threshold(
            edges, threshold, 255, cv2.THRESH_BINARY
        )[1]

    elif method == "Prewitt":
        kx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]], dtype=np.float32)
        gx = cv2.filter2D(gray, cv2.CV_32F, kx)
        gy = cv2.filter2D(gray, cv2.CV_32F, ky)
        magnitude = cv2.magnitude(gx, gy)
        edges = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        edges = cv2.threshold(
            edges, threshold, 255, cv2.THRESH_BINARY
        )[1]

    elif method == "Roberts":
        kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        gx = cv2.filter2D(gray, cv2.CV_32F, kx)
        gy = cv2.filter2D(gray, cv2.CV_32F, ky)
        magnitude = cv2.magnitude(gx, gy)
        edges = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        edges = cv2.threshold(
            edges, threshold, 255, cv2.THRESH_BINARY
        )[1]

    elif method == "Laplacian":
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        magnitude = np.abs(lap)
        edges = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        edges = cv2.threshold(
            edges, threshold, 255, cv2.THRESH_BINARY
        )[1]

    elif method == "Canny":
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        low = max(1, threshold)
        high = min(255, max(low + 1, low * 3))
        edges = cv2.Canny(blurred, low, high)

    else:
        raise ValueError("Unsupported edge detection method")

    return edges.astype(np.uint8)
