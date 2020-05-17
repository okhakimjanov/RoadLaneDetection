# Road Lane Detection using OpenCV and Python

The goal of the project is to analyze the target video and write a program to understand the movement of the car and predict the turn based on the number of techniques and algorithms applied. 

The functionality of the proposed system includes a range of image processing manipulations with images and video material:
  -  The initial processing requires a conversion of the colored image to a grayscale one.
  -  Applying a Gaussian blur for denoising the picture
  - Canny Edge Detection is applied to get the image edges
  - HoughLines Transformation for completing the lines on an image

# Expected output video:

[![alt text](https://github.com/okhakimjanov/RoadLaneDetection/blob/master/Screenshots/1024x768.png?raw=true)](https://www.youtube.com/watch?v=wx3EbhdhwVg)

<iframe width="560" height="315" src="https://www.youtube.com/embed/wx3EbhdhwVg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

`Click on the image above to see the full video on YouTube`

**Disclaimer**: There are a number of improvements to the system that could be done later. First of all, instead of straight lines, it is better to use a more complex curve, which will be useful on curved sections of the road. Also, having information from previous frames available, averaging is not always a good strategy. It is better to use weighted average or priority values.

# Algorithm Explanation
The program works in the way that it takes as an input the given video and processes each frame and makes certain decisions whether the car is moving straight, to the right or to the left. In order to identify the direction of the movement, the frame goes through certain steps so that we could find the road sidelines.

The first step is to import all necessary libraries in our project such as:
- OpenCV
- Numpy
- Time library to make time manipulations

```python
import cv2 as cv
import numpy as np
import time
```
Three variables `prev_left_avg`, `prev_right_avg` and `direction` describe the lines that we store just in case there were found no lines during the processing of the frame. 
In such cases, we draw the lines from the previous frame and these variables store slope and y-intercept of those lines. Direction stores the value of the direction towards which the car is moving.
```python
prev_left_avg = [0.000001, 0] # previous left line
prev_right_avg = [0.000001, 0] # previous right line
direction = "Straight" # the direction of the movement
```
The main steps that our program is taking in order to achieve the goal is to process each frame in the following manner:
- Read the frame
- Resize the frame
- Decrease the Brightness of the frame
- Extract only Yellow and White colors in the frame
- Convert the frame to Grayscale color space
- Blur the frame
- Binarize the frame
- Find the Edges in the frame
- Extract only the region of the interest in the frame
- Find Hough lines in the frame
- Calculate the final single left and right lines in the frame
- Increase the Brightness of the frame
- Predict the direction of the movement in the frame
- Draw the lines on the frame
- Display the frame

The first step is to read the frame which can be done easily by using the following OpenCV functions which help to open the video and read each frame writing the status of the process in the `ret` variable. The variable `ret` can be `True` for successfull opening and `False` in case no frame read.
The frame itself is stored in the variable `frame`.
 OpenCV function `get` helps getting special characteristics of the video such as **FPS** of the video which is **30** in our case.
 
```python
cap = cv.VideoCapture("Road.mp4") # read the video named "Road.mp4" 
fps = cap.get(cv.CAP_PROP_FPS) # get the FPS of the video
 while (cap.isOpened()):
    start_time = time.time()
    ret, frame = cap.read()
```

 The next step is to resize the frame. We were asked to test our program on different resolutions so that we could find the Frame Per Second value in each case. The idea of calculating the frame is quite simple: we find the total amount of time required to process the frame, and divide 1 by this period which gives us the number of frames per second.
 The Time library helps us to complete this task. The results of this mission are provided below:
**The average FPS of the video with resolution 1024x768 is 45.**
<img src="https://github.com/okhakimjanov/RoadLaneDetection/blob/master/Screenshots/1024x768.png?raw=true" width="1024">

**The average FPS of the video with resolution 800x600 is 75.**
<img src="https://github.com/okhakimjanov/RoadLaneDetection/blob/master/Screenshots/800x600.png?raw=true" width="800">

**The average FPS of the video with resolution 640x480 is 111.**
<img src="https://github.com/okhakimjanov/RoadLaneDetection/blob/master/Screenshots/640x480.png?raw=true" width="640">

**The average FPS of the video with resolution 400x300 is 166.**
<img src="https://github.com/okhakimjanov/RoadLaneDetection/blob/master/Screenshots/640x480.png?raw=true" width="400">

The next step is to darken the frame by lowering the brightness which can be done easily if we first convert the color space from `BGR` to `HSV` which opens us to change the brightness of the image by its third value. We lower the brightness of all pixels by `20` so that it becomes darker and makes all light spots of the frame more visible. Later, we revert the color space back to the `BGR`.

```python
frame = cv.resize(frame, (width, height))
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV) # BGR TO HSV to make darker
    frame[:,:,2] -= 20 # make it darker by lowering the brightness by 20
    frame = cv.cvtColor(frame, cv.COLOR_HSV2BGR) # convert back to BGR color space
```

After that we process the frame in the way that only **yellow** and **white** colors are left in it for further image analysis. This can be done by using the Masking technique which allows creating arrays of the lower and upper limits of the colors for BGR image and apply them on the frame, which makes the OpenCV leave the pixels only within the given range. We created two masks for yellow and white colors and in the end merged them into one so that we could apply both mask to the frame

```python
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

lower_white = np.array([0, 0, 100], dtype=np.uint8) # the lower limit
upper_white = np.array([255, 255, 255], dtype=np.uint8) # the upper limit

mask_white = cv.inRange(hsv, lower_white, upper_white)

lower_yellow = np.array([20, 100, 100], dtype=np.uint8) 
upper_yellow = np.array([30, 255, 255], dtype=np.uint8) 

mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
mask = cv.bitwise_or(mask_white, mask_yellow)

frame = cv.bitwise_and(frame, frame, mask=mask)
```

The next steps include the gray scaling, blurring, binarizing the frame and finding the edges of the objects detected. OpenCV library provides various functions to perform these operations: cvtColor() converts the frame to the specified color space, GaussianBlur() applies the specific blur effect with the provided kernel matrix, threshold() binarizes the image making it black and white by separating the colors by their value, Canny() allows applying the Canny Edge Detection algorithm to detect all the edges of the observed light spots. Finally, we return back the processed frame with the edges on it.

```python
def do_canny(frame):
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    binary = cv.threshold(blur, 110, 255, cv.THRESH_BINARY)[1]
    canny = cv.Canny(binary, 100, 150)
    
    return canny
```

The next step involves the core of the program which is identifying the lines from the edges, which can be performed by the Hough Lines which is implemented in the OpenCV library and returns the array of all possible lines which were detected by the algorithm. We used the advanced version of the algorithm which involves the Probabilistic approach and helps to set the limits for the Line length and the gap between the lines, extracting only the lines which are more likely to be a road line in the ROI.

```python
hough = cv.HoughLinesP(segment, 1, np.pi / 180, 25, np.array([]), minLineLength = 25, maxLineGap = 50)
```

The obtained lines are then processed to separate the left and right ones. This can be achieved by finding the slope of the lines which can be obtained from using the NumPy function “polyfit” which takes the coordinates as an input and returns the slope and y-intercept of the possible line. If the slope is less than 0 it means that the line is on the left side, otherwise it is the right road sideline. After identifying all the left and right lines by the slope, we should find the only one which could represent the left and right line. For this purpose, we decided to use the NumPy “average” function which finds the average values for the given set of elements that are slopes and y-intercepts of the lines in our case. Following this, we look at the obtained line,if it is not a line but a NaN or any other except array, this means the line was not found and therefore, we decided to demonstrate the previous line for such cases. The same logic is applied for left and right lines as each time we find a line we store it in the global variables so that we always have a left and right line which can be shown if no line is found. Such cases occur quite frequently as the line detection is quite sensitive to the values that we pass, as well as the quality of the frame, light sources and the color of the road. Finally, we calculate the coordinates of the lines and return the array of the left and right lines.

```python
def calculate_lines(frame, lines):
    global prev_left_avg, prev_right_avg

    if lines is None:
        return frame
    left = []
    right = []
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        if slope < 0:
            left.append((slope, y_intercept)) # left line
        else:
            right.append((slope, y_intercept)) # right line
    # Find average of the left and right lines into a single slope and y-intercept value
    left_avg = np.average(left, axis = 0) # the slope
    right_avg = np.average(right, axis = 0) # the y-incercept
    
    if type(left_avg) == np.ndarray:
        prev_left_avg = left_avg
    else:
        left_avg = prev_left_avg
        
    # if the right line is found then make it as the last normal right line
    if type(right_avg) == np.ndarray:
        prev_right_avg = right_avg
    else: # otherwise the previous right line is assiged to the right line
        right_avg = prev_right_avg
        
    # Find the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    
    return np.array([left_line, right_line])
```

The coordinates are easily calculated by having the initial point and then finding the other by using the main formula of the line which is `y=slope * x + y-intercept`. In the end of the function, we obtain the starting and the ending points of the line which we return as the output of the function.

```python
def calculate_coordinates(frame, line_params):
    # extract the slope and y-intercept from the line
    try:
        slope, intercept = line_params
    except:
        slope, intercept = 0, 0
   
    # if there is no line, set the points to be the bottom-center of the frame
    if slope == 0:
        y1 = frame.shape[0]
        y2 = frame.shape[0]
        x1 = frame.shape[1] / 2
        x2 = frame.shape[1] / 2
    else: # otherwise the coordinates are calculated using the formula y = x*slope + y-intercept
        y1 = frame.shape[0]
        # ending y-coordinate is 20% of the height from the bottom 
        y2 = int(y1 - frame.shape[0]*0.2)
        # starting x-coordinate is (y1 - y-intecept) / slope since y1 = slope * x1 + y-intercept
        x1 = int((y1 - intercept) / slope)
        # ending x-coordinate is (y2 - y-intercept) / slope since y2 = slope * x2 + y-intercept
        x2 = int((y2 - intercept) / slope)
    
    # return the coordinates
    return np.array([x1, y1, x2, y2])
```

Following this step, we define the direction of the road lines. In order to find the correct direction, we need to find the angle at which each of the lines are looking at. After certain amount of calculations and observations we found out that corresponding angles shown in the image on the right. Finally, we return the direction as the output of the function.

```python
def find_direction(lines):
    global direction # allow to use the global variable
    slopes = [] # array to store the available slopes of the lines
    
    # if there are no lines, return the empty frame without any changes 
    if lines is None:
        return direction
    
    # if in the case the line of the array contains more than 2 coordinates
    # return the latest direction
    if len(lines) > 2:
        return direction
    
    # for each set of coordinates, find the slope in degrees
    for x1, y1, x2, y2 in lines.reshape(2, 4):
        slopes.append(np.arctan((y2-y1) / (x2-x1)) * 180 / np.pi)
    
    # find the limitations of the slope of the left and right lines to determine the direction
    if slopes[0] < -55 and slopes[1] < 41:
        direction = "left" # left direction
    elif slopes[0] > -50 and slopes[1] > 48:
        direction = "right" # right direction
    else:
        direction = "straight" # default direction

    # return the obtained direction
    return direction
```

The last step in our program is the visualization of the lines on the frame. OpenCV provided the special function for drawing the line to which we need to give the starting and ending points as well as the color and thickness of the lines. We also made the decision to draw the polygons between the lines which gives more visual representation of the road the car is passing. We use “fillPoly” function which takes as input the points which restrict the rectangle. We return the frame with lines and rectangle drawn on it.

```python
def visualize_lines(frame, lines):
    # Create an empty image of the same size as the frame filled with the zeros
    lines_visualize = np.zeros_like(frame)
    f_points = [] # find the points for drawing the colored polygon
    
    # Create an empty image of the same size as the frame filled with the zeros
    mask = np.zeros_like(frame)
    
    # if there are no lines, return the empty frame without any changes 
    if lines is None:
        return frame
    
    # loop through each set of coordinates as the lines 
    for x1, y1, x2, y2 in lines:
        # Draws lines between two coordinates with green color and 5 thickness
        cv.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
        # append the points of the lines for the rectangle
        f_points.append((x2, y2)) 
        f_points.append((x1, y1))

    # create an empty array to rearrange the points of the polygon 
    points = []
    points.append(f_points[0]) 
    points.append(f_points[1])
    points.append(f_points[3])
    points.append(f_points[2])

    # draw the polygon with the given points with filled with red color on the mask
    cv.fillPoly(mask, np.array([points]), (0, 0, 255))
    
    # return the frame filled the drawn lines and the polygon on it
    return cv.bitwise_or(lines_visualize, mask)
```

This part of the program displays the text of the direction on the screen by providing all the styles which should be applied to the text such as font, font scale, color and thickness.
```python
    text = "DIRECTION: {}".format(direction.upper())
    org = (10, 25)
    fontFace = cv.FONT_HERSHEY_PLAIN
    fontScale = 1
    color = (0, 0, 255)
    thickness = 1
```
