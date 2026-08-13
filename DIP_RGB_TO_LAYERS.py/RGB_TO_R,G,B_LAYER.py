import cv2
import matplotlib.pyplot as plt
# -------------------------------
# Read RGB image
# -------------------------------
image = cv2.imread(r'E:\alldownloads\Projects\DIP\DIP_RGB_TO_LAYERS.py\images.jpg')
# OpenCV reads images as BGR
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# -------------------------------
# Extract RGB layers
# -------------------------------
red_layer = image_rgb[:, :, 0]
green_layer = image_rgb[:, :, 1]
blue_layer = image_rgb[:, :, 2]
# -------------------------------
# Display all layers
# -------------------------------
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original RGB Image")
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(red_layer, cmap="gray")
plt.title("Red Layer")
plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(green_layer, cmap="gray")
plt.title("Green Layer")
plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(blue_layer, cmap="gray")
plt.title("Blue Layer")
plt.axis("off")
plt.tight_layout()
plt.show()
# -------------------------------
# Save individual layers
# -------------------------------
cv2.imwrite(r'E:\alldownloads\Projects\DIP\DIP_RGB_TO_LAYERS.py\red_layer.png', red_layer)
cv2.imwrite(r'E:\alldownloads\Projects\DIP\DIP_RGB_TO_LAYERS.py\green_layer.png', green_layer)
cv2.imwrite(r'E:\alldownloads\Projects\DIP\DIP_RGB_TO_LAYERS.py\blue_layer.png', blue_layer)
print("Done!")
print("Saved:")
print("  red_layer.png")
print("  green_layer.png")
print("  blue_layer.png")