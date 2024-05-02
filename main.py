import cv2
import numpy as np

# Load images
img1 = cv2.imread('pgms4SQ24-3/A.pgm', cv2.IMREAD_ANYDEPTH)
img2 = cv2.imread('pgms4SQ24-3/B.pgm', cv2.IMREAD_ANYDEPTH)

# Convert images to uint8 data type
img1 = np.asarray(img1, dtype=np.uint8)
img2 = np.asarray(img2, dtype=np.uint8)

# Convert images to grayscale
img1_gray = img1.copy()
img2_gray = img2.copy()

# Calculate absolute difference between the two images
diff = cv2.absdiff(img1_gray, img2_gray)

# Apply threshold to identify significant differences
thresh = 30
diff[diff < thresh] = 0
diff[diff >= thresh] = 255

# Find contours of significant differences
contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw rectangles around the differences
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the output image with differences highlighted
cv2.imshow('Output', img1)
cv2.imwrite('output.png', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
