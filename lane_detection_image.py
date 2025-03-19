import cv2
import numpy as np
import math


# image pre-processing
def preprocess_and_cannify(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
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
        if math.fabs(slope) < 0.4:
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

# function to draw hough lines
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return line_image

# function to mask the image to draw/show hough lines only on required areas
def region_of_interest(image):
    height = image.shape[0]
    polygon = np.array(
        [[
                (150, height),
                (550, int(height / 2)),
                (620, int(height / 2)),
                (850, height),
        ]]
    )
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


image = cv2.imread("/home/arko/Mini_Project/test_images/urban_marked/um_000001.png")

canny_image = preprocess_and_cannify(image)

cropped_image = region_of_interest(canny_image)

lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 80, np.array([]), minLineLength=10, maxLineGap=5)
# print(lines)
averaged_lines = average_slope_intercept(image, lines)
# print(averaged_lines)
line_image = display_lines(image, averaged_lines)
# line_image = display_lines(image, lines)
combo_image = cv2.addWeighted(image, 0.9, line_image, 1, 1)

cv2.imshow("result", combo_image)
if cv2.waitKey(0):
    cv2.destroyAllWindows()
