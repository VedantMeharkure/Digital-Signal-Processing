import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


# ============================================================
# PARAMETERS
# ============================================================

INPUT_IMAGE = "input.jpg"

# Bit planes to display
# 0 = LSB
# 7 = MSB

START_BIT = 0
END_BIT = 7


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

img = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

if img is None:

    print("Image not found!")

    print(
        "Looking for:",
        image_path
    )

    exit()


print("Image loaded successfully!")

print(
    "Image size:",
    img.shape
)


# ============================================================
# BIT PLANE SLICING
# ============================================================

bit_planes = []

for bit in range(
    START_BIT,
    END_BIT + 1
):

    # Extract the particular bit
    plane = np.bitwise_and(
        img,
        1 << bit
    )

    # Convert 0/1 representation to
    # 0/255 representation for display

    plane = plane * 255

    bit_planes.append(
        plane.astype(np.uint8)
    )


# ============================================================
# DISPLAY ORIGINAL IMAGE
# ============================================================

plt.figure(
    figsize=(12, 10)
)

plt.subplot(
    3,
    3,
    1
)

plt.imshow(
    img,
    cmap="gray"
)

plt.title(
    "Original Image"
)

plt.axis("off")


# ============================================================
# DISPLAY BIT PLANES
# ============================================================

for i, plane in enumerate(
    bit_planes
):

    plt.subplot(
        3,
        3,
        i + 2
    )

    plt.imshow(
        plane,
        cmap="gray"
    )

    plt.title(
        f"Bit Plane {START_BIT + i}"
    )

    plt.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# SAVE BIT PLANES
# ============================================================

for i, plane in enumerate(
    bit_planes
):

    bit_number = START_BIT + i

    output_path = os.path.join(
        folder,
        f"bit_plane_{bit_number}.png"
    )

    cv2.imwrite(
        output_path,
        plane
    )


print()
print(
    "======================================"
)

print(
    "BIT PLANE SLICING COMPLETED"
)

print(
    "======================================"
)

for bit in range(
    START_BIT,
    END_BIT + 1
):

    print(
        f"Bit Plane {bit}: "
        f"bit_plane_{bit}.png"
    )