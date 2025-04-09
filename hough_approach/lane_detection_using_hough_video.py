import cv2
# import matplotlib.pyplot as plt
import numpy as np
import math


# frame preprocessing
def preprocess_and_cannify(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # conversion to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # using Gaussian blurring
    edges = cv2.Canny(blur, 50, 150)  # using the canny function to obtain the canny edges
    return edges


# helper function to plot the points of the averaged lines
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

# function to find the averaged hough lines for each lane
def average_slope_intercept(image, lines):
    left_lane_line = []
    right_lane_line = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameter = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameter[0]
        intercept = parameter[1]
        if math.fabs(slope) < 0.1:
            continue
        
        if slope < 0:
            left_lane_line.append((slope, intercept))
        else:
            right_lane_line.append((slope, intercept))
            
    left_fit_average = np.average(left_lane_line, axis=0)
    right_fit_average = np.average(right_lane_line, axis=0)
    
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    
    return np.array([left_line, right_line])


def display_lines(frame, lines):
    line_frame = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return line_frame
    # print(line)


def region_of_interest(edges):
    height, width = edges.shape
    polygon = np.array(
        [[
                (350, height),
                (570, int(height / 2) + 50),
                (580, int(height / 2) + 50),
                (1050, height),
        ]]
    )
    mask = np.zeros_like(edges)  # A blank mask of the same size as the edges
    cv2.fillPoly(mask, polygon, 255)  # fill the polygon on the mask
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges


# video processing
v = "media/test.mp4"
cap = cv2.VideoCapture(v)

# looping through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    edges = preprocess_and_cannify(frame)
    cropped_frame = region_of_interest(edges)
    
    lines = cv2.HoughLinesP(cropped_frame, 2, np.pi/180, 80, np.array([]), minLineLength=10, maxLineGap=40)
    # averaged_lines = average_slope_intercept(frame, lines)
    line_frame = display_lines(frame, lines)
    
    # Displaying the frames
    # cv2.imshow("Video Playback",masked_edges)
    cv2.imshow("Video Playback", line_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
