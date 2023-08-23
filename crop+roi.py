import cv2
import numpy as np
#import pandas as pd

net= cv2.dnn.readNet('15_model/yolov4-15_30000.weights','15_model/yolov4-15-30000.cfg')


net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#save all the names in file o the list classes
classes = []
with open("15_model/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
#get layers of the network
layer_names = net.getLayerNames()

#Determine the output layer names from the YOLO model 
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Capture frame-by-frame
img = cv2.imread("data/2.jpg")
# img = cv2.imread("data/now-3.jpg")
# img = cv2.imread("data/me.png")
# img = cv2.imread("data/ (1).jpeg")
# img = cv2.resize(img, None, fx=0.6, fy=0.6)
# img = cv2.resize(img,None,fx=0.4,fy=0.4, interpolation = cv2.INTER_CUBIC)
height, width, channels = img.shape

# cv2.imshow("test",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Using blob function of opencv to preprocess image
blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)
#Detecting objects
net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

#We use NMS function in opencv to perform Non-maximum Suppression
#we give it score threshold and nms threshold as arguments.
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# print(boxes)

obj = []
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        # print(i)
        label = str(classes[class_ids[i]])
    
        obj.append([label,x,y,x + w, y + h ,(2*x+w)/2,(2*y+h)/2])
        
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,
            1/2, color, 2)

# print(label)

###CROP + region_of_interest
for i in range(len(obj)):
    
    if i[0] in classes:
        
        crop = img[int(y):int(y + h),int(x):int(x + w)] 
        
        print(img.shape)
        
        height = img.shape[0]
        width = img.shape[1]
        
        region_of_interest_vertices = [
            (x, y),(x, y+h),(x+w, h),(x + w, y + h)
          ]
        
        def region_of_interest(img, vertices):
            mask = np.zeros_like(img)
            channel_count = img.shape[2]
            match_mask_color = (255,) * channel_count
            cv2.fillPoly(mask, vertices, match_mask_color)
            masked_image = cv2.bitwise_and(img, mask)
            return masked_image
        
        cropped_image = region_of_interest(img,
                        np.array([region_of_interest_vertices], np.int32),)



# img_r = cv2.resize(img, (960, 540))
img_roi = cv2.resize(crop, (960, 540))
cv2.imshow("Object_Detect_By_Image",img_roi)

###---------------------------

# img_crop = cv2.resize(cropped_image, (960, 540))
# cv2.imshow("Crop_image",img_crop)
# cv2.imshow("Object_Detect_By_Image",img)
# cv2.imshow("region_of_interest",img_r)
cv2.waitKey(0)
cv2.destroyAllWindows()
