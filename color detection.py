# Python code for Multiple Color Detection 

import numpy as np 
import cv2 


# Capturing video through webcam 
webcam = cv2.VideoCapture('data/tra1.mp4') 

# Start a while loop 
while(1): 
	
	# Reading the video from the 
	# webcam in image frames 
	_, img = webcam.read() 

	# Convert the img in 
	# BGR(RGB color space) to 
	# HSV(hue-saturation-value) 
	# color space 
	hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

	# Set range for red color and 
	# define mask 
	red_lower = np.array([136, 87, 111], np.uint8) 
	red_upper = np.array([180, 255, 255], np.uint8) 
	red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

	# Set range for green color and 
	# define mask 
    #green_lower = np.array([25, 52, 72], np.uint8) 
    #green_upper = np.array([102, 255, 255], np.uint8) 
    #green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

	green_lower = np.array([40,50,50], np.uint8) 
	green_upper = np.array([90,255,255], np.uint8) 
	green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

	# Set range for Yellow color and 
	# define mask 
	yellow_lower = np.array([15,150,150], np.uint8) 
	yellow_upper = np.array([35,255,255], np.uint8) 
	yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper) 
	
	# Morphological Transform, Dilation 
	# for each color and bitwise_and operator 
	# between img and mask determines 
	# to detect only that particular color 
	kernal = np.ones((5, 5), "uint8") 
	
	# For red color 
	red_mask = cv2.dilate(red_mask, kernal) 
	res_red = cv2.bitwise_and(img, img, 
							mask = red_mask) 
	
	# For green color 
	green_mask = cv2.dilate(green_mask, kernal) 
	res_green = cv2.bitwise_and(img, img, 
								mask = green_mask) 
	
	# For yellow color 
	yellow_mask = cv2.dilate(yellow_mask, kernal) 
	res_blue = cv2.bitwise_and(img, img, 
							mask = yellow_mask) 

	# Creating contour to track red color 
	contours, hierarchy = cv2.findContours(red_mask, 
										cv2.RETR_TREE, 
										cv2.CHAIN_APPROX_SIMPLE) 
	
	for pic, contour in enumerate(contours): 
		area = cv2.contourArea(contour) 
		if(area > 300): 
			x, y, w, h = cv2.boundingRect(contour) 
			img = cv2.rectangle(img, (x, y), 
									(x + w, y + h), 
									(0, 0, 255), 2) 
			
			cv2.putText(img, "Red Light", (x, y), 
						cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
						(0, 0, 255))	 

	# Creating contour to track green color 
	contours, hierarchy = cv2.findContours(green_mask, 
										cv2.RETR_TREE, 
										cv2.CHAIN_APPROX_SIMPLE) 
	
	for pic, contour in enumerate(contours): 
		area = cv2.contourArea(contour) 
		if(area > 300): 
			x, y, w, h = cv2.boundingRect(contour) 
			img = cv2.rectangle(img, (x, y), 
									(x + w, y + h), 
									(0, 255, 0), 2) 
			
			cv2.putText(img, "Green Light", (x, y), 
						cv2.FONT_HERSHEY_SIMPLEX, 
						1.5, (0, 255, 0)) 

	# Creating contour to track blue color 
	contours, hierarchy = cv2.findContours(yellow_mask, 
										cv2.RETR_TREE, 
										cv2.CHAIN_APPROX_SIMPLE) 
	for pic, contour in enumerate(contours): 
		area = cv2.contourArea(contour) 
		if(area > 300): 
			x, y, w, h = cv2.boundingRect(contour) 
			img = cv2.rectangle(img, (x, y), 
									(x + w, y + h), 
									(0, 255, 255), 2) 
			
			cv2.putText(img, "Yellow Light", (x, y), 
						cv2.FONT_HERSHEY_SIMPLEX, 
						1.0, (0, 255, 255)) 
			
	# Program Termination 
	cv2.imshow("Multiple Color Detection", img) 
	if cv2.waitKey(10) & 0xFF == ord('q'): 
		cv2.destroyAllWindows() 
		break
