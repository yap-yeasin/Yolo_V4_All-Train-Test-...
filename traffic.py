import cv2
import numpy as np

# image = cv2.imread("data/red.jpg")
image = cv2.imread("data/3.jpg")
# image = cv2.imread("data/yellow.jpg")
# image = cv2.imread("data/1.png")                  
# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# define Red color range
# low_red = np.array([10, 150, 80])
# high_red = np.array([180, 255, 255])

# low_red = np.array([10,100,20])
# high_red = np.array([25,255,255])

low_red = np.array([169,100,100])
high_red = np.array([189,255,255])

# low_red = np.array([100,10,0])
# high_red = np.array([0,99,100])

# #Green color
low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, low_red, high_red)

# mask = cv2.inRange(hsv, low_green, high_green)

print(type(mask))
print(mask)
# Bitwise-AND mask and original image
output = cv2.bitwise_and(image,image, mask= mask)

######################################

# from colorthief import ColorThief
# from PIL import Image

# im = Image.fromarray(output)

# # ColorThief = ColorThief('data/red.jpg')
# ColorThief = ColorThief(im)

# dominationcolor = ColorThief.get_color(quality=1)
# print(dominationcolor)

# palette = ColorThief.get_palette(quality=3)
# print(palette)

    
###################################

# cv2.imshow("Color Detected", np.hstack((image,output)))
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

