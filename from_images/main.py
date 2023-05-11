import cv2
import numpy as np
import shutil

# Read the input image
img = cv2.imread('E:\\1 number plate source\\car2.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection to the image
canny = cv2.Canny(blur, 50, 150)

# Find contours in the image
contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter the contours based on area, width, and height
for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    if (1000 < area < 50000) and (2.5 < aspect_ratio < 6):
        # Crop the region of interest from the input image
        plate = img[y:y+h, x:x+w]

        # Save the number plate as a separate image
        cv2.imwrite('number_plate.jpg', plate)

        # Draw a bounding box around the number plate
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the output image
cv2.imshow('output_image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Copy the output image to a specified folder
output_folder = 'E:\\1 number plate source\\output\\'
output_file = 'number_plate.jpg'
shutil.copy2(output_file, output_folder)
