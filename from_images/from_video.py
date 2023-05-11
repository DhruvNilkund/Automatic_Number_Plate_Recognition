import cv2
import os

# Load the video file
video = cv2.VideoCapture("E:\\1 number plate source\\video.mp4")

# Load the classifier for car detection
car_cascade = cv2.CascadeClassifier("car.xml")

# Create a folder to store the car images
if not os.path.exists("cars"):
    os.makedirs("cars")

# Initialize a counter for the number of car images extracted
count = 0

# Loop through each frame in the video
while True:
    # Read a frame from the video
    ret, frame = video.read()

    # If the frame was not successfully read, break out of the loop
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame using the classifier
    cars = car_cascade.detectMultiScale(gray, 1.1, 5)

    # Loop through each detected car and extract it as an image
    for (x, y, w, h) in cars:
        car_img = frame[y:y+h, x:x+w]
        cv2.imwrite(f"cars/car{count}.png", car_img)
        count += 1
# Release the video object
video.release()
