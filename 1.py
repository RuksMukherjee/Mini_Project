import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    if lines is None:  # If no lines are detected
        return []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # Check if left_fit and right_fit are non-empty
    left_line = None
    right_line = None

    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)

    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)

    lines = [line for line in (left_line, right_line) if line is not None]
    return np.array(lines)

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygon = np.array([
        [(375, height), (720, height), (365, 221)]
    ], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Load the image
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

# Canny edge detection
canny_image = canny(image)

# Cropping the region of interest
cropped_image = region_of_interest(canny_image)

# Hough Line Transform
lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

# Average lines
averaged_lines = average_slope_intercept(lane_image, lines)

# Draw lines
line_image = display_lines(lane_image, averaged_lines)

# Combine line image with the original
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

# Display the result
cv2.imshow("result",combo_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
