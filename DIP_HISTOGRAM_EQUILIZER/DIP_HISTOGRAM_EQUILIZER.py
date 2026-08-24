import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# PARAMETERS
# ============================================================

# Input image name
INPUT_IMAGE = "input.jpg"

# ----------------------------
# Local Histogram Equalization
# ----------------------------

# Size of local neighborhood
# Must be an odd number
LOCAL_WINDOW_SIZE = 51


# ----------------------------
# Adaptive Histogram Equalization
# CLAHE parameters
# ----------------------------

# Controls amount of contrast enhancement
CLAHE_CLIP_LIMIT = 2.0

# Size of local tiles
CLAHE_TILE_SIZE = (8, 8)


# ============================================================
# FIND IMAGE IN SAME FOLDER
# ============================================================

folder = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(
    folder,
    INPUT_IMAGE
)


# ============================================================
# READ IMAGE
# ============================================================

img = cv2.imread(image_path)

if img is None:
    print("Image not found!")
    print("Looking for:")
    print(image_path)
    exit()


print("Image loaded successfully!")


# ============================================================
# CHECK IMAGE TYPE
# ============================================================

if len(img.shape) == 2:

    # Image is already grayscale

    original = img

    color_image = False

else:

    # Image is RGB / Color

    color_image = True

    # OpenCV reads BGR
    rgb_image = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # Convert BGR -> YCrCb
    ycrcb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2YCrCb
    )

    # Extract Y channel
    # Y = brightness
    # Cr and Cb = color information

    original = ycrcb[:, :, 0]


# ============================================================
# GLOBAL HISTOGRAM EQUALIZATION
# ============================================================

print("Performing Global Histogram Equalization...")

global_output = cv2.equalizeHist(
    original
)


# ============================================================
# LOCAL HISTOGRAM EQUALIZATION
# ============================================================

def local_histogram_equalization(
        image,
        window_size):

    # Window size must be odd
    if window_size % 2 == 0:
        window_size += 1

    # Padding
    padding = window_size // 2

    padded_image = cv2.copyMakeBorder(
        image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_REFLECT
    )

    # Create output image
    output = np.zeros_like(image)

    height, width = image.shape

    # Process every pixel
    for i in range(height):

        for j in range(width):

            # Get local window
            window = padded_image[
                i:i + window_size,
                j:j + window_size
            ]

            # Calculate histogram
            histogram = np.bincount(
                window.ravel(),
                minlength=256
            )

            # Calculate CDF
            cdf = histogram.cumsum()

            # Find first non-zero CDF value
            non_zero = cdf[cdf > 0]

            if len(non_zero) == 0:
                output[i, j] = image[i, j]
                continue

            cdf_min = non_zero[0]

            # Number of pixels
            total_pixels = window.size

            # Current pixel
            pixel_value = image[i, j]

            # Histogram equalization formula
            denominator = (
                total_pixels - cdf_min
            )

            if denominator == 0:

                new_value = pixel_value

            else:

                new_value = (
                    (cdf[pixel_value] - cdf_min)
                    * 255
                    / denominator
                )

            # Keep value between 0 and 255
            output[i, j] = np.clip(
                new_value,
                0,
                255
            )

    return output.astype(
        np.uint8
    )


print(
    "Performing Local Histogram Equalization..."
)

print(
    "Local Window Size:",
    LOCAL_WINDOW_SIZE
)

local_output = local_histogram_equalization(
    original,
    LOCAL_WINDOW_SIZE
)


# ============================================================
# ADAPTIVE HISTOGRAM EQUALIZATION
# CLAHE
# ============================================================

print(
    "Performing Adaptive Histogram Equalization..."
)

print(
    "CLAHE Clip Limit:",
    CLAHE_CLIP_LIMIT
)

print(
    "CLAHE Tile Size:",
    CLAHE_TILE_SIZE
)


clahe = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP_LIMIT,
    tileGridSize=CLAHE_TILE_SIZE
)

adaptive_output = clahe.apply(
    original
)


# ============================================================
# FUNCTION TO CONVERT Y CHANNEL BACK TO COLOR
# ============================================================

def convert_to_display(equalized_image):

    if not color_image:

        return equalized_image

    # Keep original Cr and Cb
    result_ycrcb = cv2.merge(
        (
            equalized_image,
            ycrcb[:, :, 1],
            ycrcb[:, :, 2]
        )
    )

    # YCrCb -> BGR
    result_bgr = cv2.cvtColor(
        result_ycrcb,
        cv2.COLOR_YCrCb2BGR
    )

    # BGR -> RGB
    result_rgb = cv2.cvtColor(
        result_bgr,
        cv2.COLOR_BGR2RGB
    )

    return result_rgb


# Convert outputs for display

global_display = convert_to_display(
    global_output
)

local_display = convert_to_display(
    local_output
)

adaptive_display = convert_to_display(
    adaptive_output
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

plt.figure(
    figsize=(16, 12)
)


# ============================================================
# INPUT IMAGE
# ============================================================

plt.subplot(4, 4, 1)

if color_image:

    plt.imshow(
        rgb_image
    )

else:

    plt.imshow(
        original,
        cmap="gray"
    )

plt.title(
    "Input Image"
)

plt.axis("off")


# ============================================================
# INPUT HISTOGRAM
# ============================================================

plt.subplot(4, 4, 2)

plt.hist(
    original.ravel(),
    bins=256,
    range=[0, 256]
)

plt.title(
    "Input Histogram"
)

plt.xlabel(
    "Pixel Intensity"
)

plt.ylabel(
    "Frequency"
)


# ============================================================
# GLOBAL OUTPUT
# ============================================================

plt.subplot(4, 4, 5)

if color_image:

    plt.imshow(
        global_display
    )

else:

    plt.imshow(
        global_output,
        cmap="gray"
    )

plt.title(
    "Global Histogram Equalization"
)

plt.axis("off")


# ============================================================
# GLOBAL HISTOGRAM
# ============================================================

plt.subplot(4, 4, 6)

plt.hist(
    global_output.ravel(),
    bins=256,
    range=[0, 256]
)

plt.title(
    "Global HE Histogram"
)

plt.xlabel(
    "Pixel Intensity"
)

plt.ylabel(
    "Frequency"
)


# ============================================================
# LOCAL OUTPUT
# ============================================================

plt.subplot(4, 4, 9)

if color_image:

    plt.imshow(
        local_display
    )

else:

    plt.imshow(
        local_output,
        cmap="gray"
    )

plt.title(
    "Local Histogram Equalization"
)

plt.axis("off")


# ============================================================
# LOCAL HISTOGRAM
# ============================================================

plt.subplot(4, 4, 10)

plt.hist(
    local_output.ravel(),
    bins=256,
    range=[0, 256]
)

plt.title(
    "Local HE Histogram"
)

plt.xlabel(
    "Pixel Intensity"
)

plt.ylabel(
    "Frequency"
)


# ============================================================
# ADAPTIVE OUTPUT
# ============================================================

plt.subplot(4, 4, 13)

if color_image:

    plt.imshow(
        adaptive_display
    )

else:

    plt.imshow(
        adaptive_output,
        cmap="gray"
    )

plt.title(
    "Adaptive HE (CLAHE)"
)

plt.axis("off")


# ============================================================
# ADAPTIVE HISTOGRAM
# ============================================================

plt.subplot(4, 4, 14)

plt.hist(
    adaptive_output.ravel(),
    bins=256,
    range=[0, 256]
)

plt.title(
    "Adaptive HE Histogram"
)

plt.xlabel(
    "Pixel Intensity"
)

plt.ylabel(
    "Frequency"
)


# ============================================================
# SHOW RESULT
# ============================================================

plt.tight_layout()

plt.show()


# ============================================================
# SAVE OUTPUT IMAGES
# ============================================================

if color_image:

    # RGB -> BGR before saving

    global_save = cv2.cvtColor(
        global_display,
        cv2.COLOR_RGB2BGR
    )

    local_save = cv2.cvtColor(
        local_display,
        cv2.COLOR_RGB2BGR
    )

    adaptive_save = cv2.cvtColor(
        adaptive_display,
        cv2.COLOR_RGB2BGR
    )

else:

    global_save = global_output

    local_save = local_output

    adaptive_save = adaptive_output


# Global output
cv2.imwrite(
    os.path.join(
        folder,
        "global_output.jpg"
    ),
    global_save
)


# Local output
cv2.imwrite(
    os.path.join(
        folder,
        "local_output.jpg"
    ),
    local_save
)


# Adaptive output
cv2.imwrite(
    os.path.join(
        folder,
        "adaptive_output.jpg"
    ),
    adaptive_save
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("==========================================")
print(" HISTOGRAM EQUALIZATION COMPLETED")
print("==========================================")

print(
    "Input Image      :",
    INPUT_IMAGE
)

print(
    "Local Window     :",
    LOCAL_WINDOW_SIZE
)

print(
    "CLAHE Clip Limit :",
    CLAHE_CLIP_LIMIT
)

print(
    "CLAHE Tile Size  :",
    CLAHE_TILE_SIZE
)

print()
print(
    "Global Output    : global_output.jpg"
)

print(
    "Local Output     : local_output.jpg"
)

print(
    "Adaptive Output  : adaptive_output.jpg"
)

print("==========================================")