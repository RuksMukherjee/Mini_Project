import cv2
import matplotlib.pyplot as plt
import numpy as np

def canny(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) #conversion to grayscale
    blur=cv2.GaussianBlur(gray,(5,5),0)       #using Gaussian blurring
    edges=cv2.Canny(blur,50,200)              #using the canny function to obtain the canny edges
    return edges
def display_lines(frame, lines):
    line_frame=np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2=line.reshape(4)
            cv2.line(line_frame, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_frame
            # print(line)

def region_of_interest(edges):
    height,width=edges.shape
    polygon=np.array([
    [(0,height),(1200,height),(570,420)]
    ],dtype=np.int32)
    mask=np.zeros_like(edges) #A blank mask of the same size as the edges
    cv2.fillPoly(mask,polygon,255) #fill the polygon on the mask
    masked_edges=cv2.bitwise_and(edges,mask)
    return masked_edges

# video processing
v='test.mp4'
cap=cv2.VideoCapture(v)

# looping through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    edges=canny(frame)
    masked_edges=region_of_interest(edges)
    lines=cv2.HoughLinesP(masked_edges,2, np.pi/180,100, np.array([]), minLineLength=40, maxLineGap=5)
    line_frame=display_lines(frame, lines)
# Displaying the frames
    # cv2.imshow("Video Playback",masked_edges)
    cv2.imshow("Video Playback",line_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
