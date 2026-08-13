import cv2
# Read RGB image
image = cv2.imread(r'e:\alldownloads\Projects\DIP\DIP_RGB_TO_GREY\input.jpg')
# Convert RGB/BGR image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Display original image
cv2.imshow("Original Image", image)
# Display grayscale image
cv2.imshow("Grayscale Image", gray)
# Save grayscale image
cv2.imwrite(r'e:\alldownloads\Projects\DIP\DIP_RGB_TO_GREY\grayscale.jpg', gray)
# Wait until a key is pressed
cv2.waitKey(0)
# Close all windows
cv2.destroyAllWindows()