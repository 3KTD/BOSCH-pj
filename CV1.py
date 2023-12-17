# Import Libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load image
img = cv.imread('/content/circles.jpg')

# Gray
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# Noise Filter
gray = cv.GaussianBlur(gray, (5,5), 0)

# Binary
b_img = cv.threshold(gray, 250, 255, cv.THRESH_BINARY_INV)[1]

# find Contours
contours, hierachy = cv.findContours(b_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print("Len contours:",len(contours))

# Draw
for index, c in enumerate(contours):
  (x,y), radius = cv.minEnclosingCircle(c)
  center = int(x), int(y)
  radius = int(radius)
  cv.circle(img, center, radius, (0,255,0),2)
  text = '#' + str(index+1)
  cv.putText(img, text, center, cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# Convert to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# # Show image
plt.imshow(img)
plt.show()
plt.imshow(gray, cmap='gray')
plt.show()
plt.imshow(b_img, cmap='gray')
plt.show()

# Show Contour
contour_img = np.zeros_like(img)
cv.drawContours(contour_img, contours, -1, (255,255,255),2)
plt.imshow(contour_img, cmap='gray')
plt.show()