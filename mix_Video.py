import cv2
import numpy as np
# from imutils.video import FPS
import time

net= cv2.dnn.readNet('15_model/yolov4-15_30000.weights','15_model/yolov4-15-30000.cfg')
net1 = cv2.dnn.readNet('yolov4/yolov4.weights','yolov4/yolov4.cfg')

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


### save all the names in files o the list classes
classes = []
with open("15_model/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

classes1 = []
with open("yolov4/coco.names", "r") as f:
    classes1 = [line.strip() for line in f.readlines()]
    

### Get layers of the network ###
layer_names = net.getLayerNames()
layer_names1 = net1.getLayerNames()

### Determine the output layer names from the YOLO model ###
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
output_layers1 = [layer_names1[i[0] - 1] for i in net1.getUnconnectedOutLayers()]

### Video Input ###
# video_capture = cv2.VideoCapture("data/tra1.mp4")
# video_capture = cv2.VideoCapture("data/Yeasin/4k- (4).mp4")
# video_capture = cv2.VideoCapture("data/Yeasin/Walking.mp4")

### Web Cam ###
video_capture = cv2.VideoCapture(1)


### For Saving the Output ### 
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
size = (frame_width, frame_height)

# print(frame_width)

result = cv2.VideoWriter('detect/filename.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)


while True:
    ret,img = video_capture.read()
    _,img_copy = video_capture.read()
    

    if ret == True:
        # Capture frame-by-frame
        # img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
        
        ### Object Detection Start ###
        
        start_time = time.time() # start time of the loop
    
        # USing blob function of opencv to preprocess image
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        
        #Detecting objects
        net.setInput(blob)
        net1.setInput(blob)

        outs = net.forward(output_layers)
        outs1 = net1.forward(output_layers1)
    
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
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 2, color, 3)
        
        
        # Showing informations on the screen
        class_ids1 = []
        confidences1 = []
        boxes1 = []
        for ou in outs1:
            for detection in ou:
                scores1 = detection[5:]
                class_id1 = np.argmax(scores1)
                confidence1 = scores1[class_id1]
                if confidence1 > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
    
                    boxes1.append([x, y, w, h])
                    confidences1.append(float(confidence1))
                    class_ids1.append(class_id1)
        
        #We use NMS function in opencv to perform Non-maximum Suppression
        #we give it score threshold and nms threshold as arguments.
        indexes1 = cv2.dnn.NMSBoxes(boxes1, confidences1, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(classes1), 3))
        
        obj = ['1','2','3','4']
        for i in range(len(boxes1)):
            if i in indexes1:
                x, y, w, h = boxes1[i]
                label = str(classes1[class_ids1[i]])
                color = colors[class_ids1[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 2, color, 3)        
                # cv2.putText(img, f'{label, confidence}', (x, y + 30), font, 2, color, 2)
                if 'traffic light' == label:
                    obj[0] = int(x)
                    obj[1] = int(y)
                    obj[2] = int(w)
                    obj[3] = int(h)
                    # obj.append(label)
                # print(label)
        
        print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
        
        ### Object Detection End------------------------ ###
        
        ###  Crop + ROI +++++++++++++++++++++++++++++++++###
            
        # crop = img_copy[int(y):int(y + h),int(x):int(x + w)] 
        
        crop = img_copy[int(obj[1]):int(obj[1] + obj[3]),int(obj[0]):int(obj[0] + obj[2])] 
        
        # img_roi = cv2.resize(crop, (960, 540))
        cv2.imshow("Crop_Image",crop)
        
        
        # height = img.shape[0]
        # width = img.shape[1]
        
        # region_of_interest_vertices = [
        #     (x, y),(x, y+h),(x+w, h),(x + w, y + h)
        #   ]
        
        # def region_of_interest(img, vertices):
        #     mask = np.zeros_like(img)
        #     channel_count = img.shape[2]
        #     match_mask_color = (255,) * channel_count
        #     cv2.fillPoly(mask, vertices, match_mask_color)
        #     masked_image = cv2.bitwise_and(img, mask)
        #     return masked_image
        
        # cropped_image = region_of_interest(img,
        #                 np.array([region_of_interest_vertices], np.int32),)
        
        ### END OF CROP + ROI ------------------------------------###
        
        ### Traffic light Start ++++++++++++++++++++++++++++++++++###


        hsvFrame = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) 

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
        res_red = cv2.bitwise_and(crop, crop, 
                                mask = red_mask) 
        
        # For green color 
        green_mask = cv2.dilate(green_mask, kernal) 
        res_green = cv2.bitwise_and(crop, crop, 
                                    mask = green_mask) 
        
        # For yellow color 
        yellow_mask = cv2.dilate(yellow_mask, kernal) 
        res_blue = cv2.bitwise_and(crop, crop, 
                                mask = yellow_mask) 

        # Creating contour to track red color 
        contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                # x, y, w, h = cv2.boundingRect(contour) 
                img = cv2.rectangle(img, (x, y),(x + w, y + h),(0, 0, 255), 2) 
                
                cv2.putText(img, "Red Light", (frame_width-300, 40),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255),2)     

        # Creating contour to track green color 
        contours, hierarchy = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv2.boundingRect(contour) 
                img = cv2.rectangle(img, (x, y),(x + w, y + h),(0, 255, 0), 2) 
                
                cv2.putText(img, "Green Light", (frame_width-300, 40),cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 255, 0),2) 

        # Creating contour to track blue color 
        contours, hierarchy = cv2.findContours(yellow_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv2.boundingRect(contour) 
                img = cv2.rectangle(img, (x, y),(x + w, y + h),(0, 255, 255), 2) 
                
                cv2.putText(img, "Yellow Light", (frame_width-300, 40),cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 255),2)
        
        ### Traffic light End ------------------------------------###
        
        
        ###______Final + Walkway_________________________________________#########
        
        
        
        
        ###______Final + Walkway-END_________________________________________#########        
        
        
        # cv2.imshow(img)
        result.write(img)
        
        cv2.imshow("Output",cv2.resize(img, (800,600)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video_capture.release()
result.release()
cv2.destroyAllWindows()
print("The video was successfully saved")




