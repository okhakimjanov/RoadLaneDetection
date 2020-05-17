import cv2 as cv # OpenCV library for image processing
import numpy as np # Numpy library for operations with matrices
import time # Time library for operations with time

prev_left_avg = [0.000001, 0] # previous left line
prev_right_avg = [0.000001, 0] # previous right line
direction = "Straight" # the direction of the car

# function to extract only yellow and white colors from the given frame
def extract_yellow_white(frame):
    # Apply masks for white and yellow lines
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # the limits for white color
    lower_white = np.array([0, 0, 100], dtype=np.uint8) # the lower limit
    upper_white = np.array([255, 255, 255], dtype=np.uint8) # the upper limit
    # the mask of the only white color of the frame
    mask_white = cv.inRange(hsv, lower_white, upper_white)
    
    # the limits of yellow color
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8) # the lower limit
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8) # the upper limit
    # the mask of the only yellow color of the frame
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    
    # merge both masks into one mask
    mask = cv.bitwise_or(mask_white, mask_yellow)
    # apply the masks to the frame
    frame = cv.bitwise_and(frame, frame, mask=mask)
    
    # return the frame with only white and yellow colors
    return frame

# function to find the edges of the frame 
def do_canny(frame):
    # Convert the frame to GRAYSCALE image 
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # Make the frame blurry for smoothing the corners
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    # Convert the frame to BINARY image (black and white)
    binary = cv.threshold(blur, 110, 255, cv.THRESH_BINARY)[1]
    # Apply the Canny Edge Detection in order to find the edges in the frame 
    canny = cv.Canny(binary, 100, 150)
    
    # Return the final image containing the Edges of the objects
    return canny

# function to extract only the lower part of the image
def do_segment(frame):
    # get the height and width of the frame for the later work
    height = frame.shape[0] # get the height
    width = frame.shape[1] # get the width
    # Create a polygon of 4 corners which describes the the part of the image to look at
    polygons = np.array([
                            [(5 * width//16, height), # left bottom point
                             (6 * width // 16, 7 * height//9), # left top point 
                             (10 * width // 16, 7 * height//9), # right top point
                             (14*width//16, height)] # right bottom point
                        ])
    
    # Create an empty image of the same size as the frame filled with the zeros 
    mask = np.zeros_like(frame)
    # Fill the area described with the polygons with 1s, the other parts with 0s
    cv.fillPoly(mask, polygons, 255)
    # Apply the mask to the frame in order to extract this part of the original image 
    segment = cv.bitwise_and(frame, mask)
    
    # return the obtained segment of the frame
    return segment

# function to identify the lines on the frames from Hough Lines 
def calculate_lines(frame, lines):
    global prev_left_avg, prev_right_avg # use the global variables in this function
    
    # if there are no lines, return the empty frame without any changes 
    if lines is None:
        return frame
    
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    
    # Loops through every detected line
    for line in lines:
        # Reshapes line from 2D array to 1D array of size 4
        x1, y1, x2, y2 = line.reshape(4)
        # Fit the points to a linear polynomial to return a vector of the slope and y-intercept of each line
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # extract the slope and y-intecept separately
        slope = parameters[0]
        y_intercept = parameters[1]
        # If slope is negative, the line is the left, and otherwise, the line is the right
        if slope < 0:
            left.append((slope, y_intercept)) # left line
        else:
            right.append((slope, y_intercept)) # right line
        
    # Find average of the left and right lines into a single slope and y-intercept value
    left_avg = np.average(left, axis = 0) # the slope
    right_avg = np.average(right, axis = 0) # the y-incercept
    
    # if the left line is found then make it as the last normal left line
    if type(left_avg) == np.ndarray:
        prev_left_avg = left_avg
    else: # otherwise the previous left line is assiged to the left line
        left_avg = prev_left_avg
        
    # if the right line is found then make it as the last normal right line
    if type(right_avg) == np.ndarray:
        prev_right_avg = right_avg
    else: # otherwise the previous right line is assiged to the right line
        right_avg = prev_right_avg
        
    # Find the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    
    # return the left the right line as the output of the function
    return np.array([left_line, right_line])

# Find the coordinates of the lines in the frame
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

# draw the lines on the rame
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

# function to find the direction of the car movement 
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
        
# The video feed is read in as a VideoCapture object
def videoReader(width, height):
    global direction # use the global variable
    cap = cv.VideoCapture("Road.mp4") # read the video named "Road.mp4" 
    fps = cap.get(cv.CAP_PROP_FPS) # find the FPS of the video
    
    # while the camera is openned 
    while (cap.isOpened()):
        # store the starting time before the frame processing
        start_time = time.time()
        # ret = a boolean return value from getting the frame
        # frame = the current frame being projected in the video
        ret, frame = cap.read()
        # if the ret is True which means the video is not ended
        if ret:
            try: 
                # resize the image to the given resolution (width, height)
                frame = cv.resize(frame, (width, height))
                
                # convert to HSV color space in order to make it darker
                frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                # make it darker by lowering the brightness by 20
                frame[:,:,2] -= 20
                # convert back to BGR color space
                frame = cv.cvtColor(frame, cv.COLOR_HSV2BGR)
                
                # extract only white and yellow colors from frame
                extract = extract_yellow_white(frame)
                
                # canny edge detection
                canny = do_canny(extract)

                # region of interest
                segment = do_segment(canny)
                
                # hough lines with the specific parameters
                hough = cv.HoughLinesP(segment, 1, np.pi / 180, 25, np.array([]), minLineLength = 25, maxLineGap = 50)

                # find average values of multiple detected lines from hough                
                lines = calculate_lines(frame, hough)

                # convert to HSV color space in order to make it lighter
                frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                # make it lighter by rising the brightness by 20
                frame[:,:,2] += 20
                # convert back to BGR color spaces
                frame = cv.cvtColor(frame, cv.COLOR_HSV2BGR)
                
                # Predict the direction of the car movement
                direction = find_direction(lines)
                # Drawing the lines on the frame
                lines_visualize = visualize_lines(frame, lines)

                # Overlay lines on frame
                output = cv.addWeighted(frame, 1, lines_visualize, 1, 1)
            except:
                # the case if some error occurs during the execution of the functions
                output = frame
        else:
            # the case if the video has ended
            break
    
        # store the ending time after the frame processing
        end_time = time.time()

        # determine what text to display as the first text
        # display the direction as red text with thickness of 1 and default font styles 
        text = "DIRECTION: {}".format(direction.upper())
        org = (10, 25)
        fontFace = cv.FONT_HERSHEY_PLAIN
        fontScale = 1
        color = (0, 0, 255)
        thickness = 1
        
        # put the diection text on the frame
        output = cv.putText(output, text, org, fontFace, fontScale, color, thickness)

        # determine what text to display as the second text
        # display the FPS as red text with thickness of 1 and default font styles
        # FPS can be found by division of 1 by the overall time of the frame processing  
        text = "FPS: {}".format(int(1/(end_time - start_time)))
        org = (10, 50)

        # put the FPS text on the frame
        output = cv.putText(output, text, org, fontFace, fontScale, color, thickness)

        # Opens a new window and displays the output frame
        cv.imshow("Road.mp4", output)
        # frames are read by intervals of 10 milliseconds
        # the program ends when 'q' key is pressed
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
            
    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()

# ask the user in what resolution to show the video
print("Choose the video size: ")
print("1. 1024x768")
print("2. 800x600")
print("3. 640x480")
print("4. 400x300")

option = input() # ask for the option
if option == '1':
    videoReader(width=1024, height=768)
elif option == '2':
    videoReader(width=800, height=600)
elif option == '3':
    videoReader(width=640, height=480)
elif option == '4':
    videoReader(width=400, height=300)
    