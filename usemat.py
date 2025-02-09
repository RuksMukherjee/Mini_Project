import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Conversion to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)       # Using Gaussian blurring
    edges = cv2.Canny(blur, 50, 150)               # Using the Canny function to obtain the edges
    return edges

# Video processing
v = 'test.mp4'
cap = cv2.VideoCapture(v)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set up the matplotlib figure
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Reached the end of the video or cannot fetch the frame.")
        break

    edges = canny(frame)  # Apply the Canny function

    # Display the processed frames using matplotlib
    ax.imshow(edges, cmap='gray')  # Display in grayscale
    ax.set_title("Canny Edge Detection")
    # ax.axis('off')  # Turn off axes for better visualization
    plt.pause(0.001)  # Pause briefly to simulate real-time playback
    ax.clear()        # Clear the figure for the next frame

# Release resources
cap.release()
plt.close(fig)
