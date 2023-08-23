import os
import numpy as np 
import cv2 


# def detect(filepath,file):
# def detect(picture):
    
font = cv2.FONT_HERSHEY_SIMPLEX
# img = cv2.imread(filepath+file)
# img = cv2.imread('data/1.jpg')
img = cv2.imread("data/2.jpg")

# cv2.imshow("Multiple Color Detection in Real-TIme", img) 

# cv2.waitKey(0)
# cv2.destroyAllWindows()

cimg=img
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# #Color Range
#RED
low_red = np.array([0,100,100])
up_red = np.array([10,255,255])

low_red1 = np.array([160,100,100])
up_red1 = np.array([180,255,255])

#Green
low_green = np.array([40,50,50])
up_green = np.array([90,255,255])

# low_green1 = np.array([0,100,100])
# up_green1 = np.array([10,255,255])

#Yellow
low_yellow = np.array([15,150,150])
up_yellow = np.array([35,255,255])

# low_yellow1 = np.array([0,100,100])
# up_yellow1 = np.array([10,255,255])

mask_r = cv2.inRange(hsv, low_red, up_red)
mask_r1 = cv2.inRange(hsv, low_red1, up_red1)

mask_g = cv2.inRange(hsv, low_green, up_green)

mask_y = cv2.inRange(hsv, low_yellow, up_yellow)

mask_r1_add = cv2.add( mask_r, mask_r1)


size = img.shape

# #Hough Circle detect
r_circle = cv2.HoughCircles(mask_r1_add, cv2.HOUGH_GRADIENT, 1, 80,
                            param1=50 ,param2 = 10 ,minRadius = 0, maxRadius = 30)

g_circle = cv2.HoughCircles(mask_g, cv2.HOUGH_GRADIENT, 1, 60,
                            param1=50 ,param2 = 10 ,minRadius = 0, maxRadius = 30)    

y_circle = cv2.HoughCircles(mask_y, cv2.HOUGH_GRADIENT, 1, 30,
                            param1=50 ,param2 = 5 ,minRadius = 0, maxRadius = 30)


r = 5
bound = 4.0/10

##RED_Text_section
if r_circle is not None:
    r_circle = np.uint16(np.around(r_circle))
    
    for i in r_circle[0,:]:
        if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
            continue
        
    h,s = 0.0,0.0
    for m in range(-r,r):
        for n in range(-r,r):
            if (i[1]+m) >= size[0] or (i[0]+n) >=size[1]:
                continue
            h+=mask_r1_add[i[1]+m,i[0]+n]
            s+=1
            
    if h/s >50:
        cv2.circle(cimg,(i[0],i[1]), i[2]+10,(0,255,0),2)
        cv2.circle(mask_r1_add,(i[0],i[1]), i[2]+30, (255,255,255),2)
        cv2.putText(cimg,'RED',(i[0],i[1]), font ,1,(255,0,0),2,cv2.LINE_AA )
        print('R1')

##GREEN_TEXT_SECTION
if g_circle is not None:
    g_circle = np.uint16(np.around(g_circle))
    
    for i in g_circle[0,:]:
        if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
            continue
        
    h,s = 0.0,0.0
    for m in range(-r,r):
        for n in range(-r,r):
            if (i[1]+m) >= size[0] or (i[0]+n) >=size[1]:
                continue
            h+=mask_g[i[1]+m,i[0]+n]
            s+=1
            
    if h/s >100:
        cv2.circle(cimg,(i[0],i[1]), i[2]+10,(0,255,0),2)
        cv2.circle(mask_g,(i[0],i[1]), i[2]+30, (255,255,255),2)
        cv2.putText(cimg,'GREEN',(i[0],i[1]), font ,1,(255,0,0),2,cv2.LINE_AA )
        print('G1')

##Yellow_TEXT_SECTION
if y_circle is not None:
    y_circle = np.uint16(np.around(y_circle))
    
    for i in y_circle[0,:]:
        if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
            continue
        
    h,s = 0.0,0.0
    for m in range(-r,r):
        for n in range(-r,r):
            if (i[1]+m) >= size[0] or (i[0]+n) >=size[1]:
                continue
            h+=mask_y[i[1]+m,i[0]+n]
            s+=1
            
    if h/s >50:
        cv2.circle(cimg,(i[0],i[1]), i[2]+10,(0,255,0),2)
        cv2.circle(mask_y,(i[0],i[1]), i[2]+30, (255,255,255),2)
        cv2.putText(cimg,'YELLOW',(i[0],i[1]), font ,1,(255,0,0),2,cv2.LINE_AA )
        print('Y1')
        
cv2.imshow("Detected_result", cimg)    
# cv2.imwrite(path+'//result//'+file ,cimg)


cv2.waitKey(0)
cv2.destroyAllWindows()

    
# if __name__ == '__main__':
#     # path = os.path.abspath('..')+'//light//'
    
#     path = 'data/'
    
#     for f in os.listdir(path):
#         print(f)
#         if f.endswith('.jpg') or f.endswith('.png'):
#             detect(path, f)

# path = "data/"

# for f in os.listdir(path):
#     # print(f)
#     if f.endswith('.jpg') or f.endswith('.png'):
#         detect(f)
#         print(f)
            
            
            
            
            
            
    
    