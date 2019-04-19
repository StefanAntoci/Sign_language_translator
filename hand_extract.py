import cv2
import numpy as np
import math
from os import listdir
from os.path import isfile, join
import csv
import os
import random

key_points = list()
key_points2 = list()
key_points3 = list()

rev_key_points = list()
rev_key_points2 = list()
rev_key_points3 = list()

key_points_number = 20
dists = list()
slopes =list()
conv_def = list()
idx = 0

def write_to_csv(label, key_points):    
    global idx
    #x =random.randint(0, 20)
    if idx % 50 == 0:
        for data in key_points:
            test_data = open("test_data.csv", "a")
            test_data.write(str(data) )
            test_data.write(" , ")
            
        test_data.write(label)
        test_data.write("\n")
        test_data.close()
        key_points.clear()
        
       
    else:
        for data in key_points:
            train_data = open("train_data.csv", "a")
            train_data.write(str(data) )
            train_data.write(" , ")
            
        train_data.write(label)
        train_data.write("\n")
        train_data.close()
        key_points.clear()

        
def create_feature_array(key_points, ref_point):
    for x in range(len(key_points)):
        key_points[x] = calculate_distance(key_points[x], ref_point)
    return key_points
    
def compute_center(cnt):
    moments = cv2.moments(cnt)
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    center = [cx, cy]
    return center
    
def get_max_contour(img):
    global label
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(img.shape, np.uint8)
    #print(drawing)
    max_area = 0.0
    ci = 0
    #print(len(contours))
    for i in range (len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area > max_area):
            max_area = area
            ci = i        
    
    cnt = contours[ci]
    #print(cnt)
    #cv2.drawContours(drawing, [cnt], 0, (255,255,255), 2)
    #cv2.imshow(label + "adadasdas", drawing)
    return cnt

def calculate_distance(start, end):
    dist = math.sqrt( (start[1] - end[1]) ** 2 + (start[0] - end[0]) ** 2)
    return dist
    
def get_key_points(cnt):
    key_points = list()
    key_points.append(cnt[0][0])
    dist = 0
    start_point = cnt[0][0]
    end_point = 0
    i  = 5
    pts_number = len(cnt)
    while i < pts_number:
        while dist < 10:
            if i >= pts_number:
                break
            end_point = cnt[i][0]
            dist = calculate_distance(start_point, end_point)
            i = i + 5
        if dist >= 10:
            key_points.append(end_point)
            start_point = end_point
            dist = 0
    return key_points

def delete_key_points(key_points, points_nr):
    pts_dist = dict()
    aux = list()
    for x in range(points_nr):
        for i in range(len(key_points)):
            if i == 0:
                dist1 = calculate_distance( key_points[len(key_points)-1], key_points[0] )
                dist2 = calculate_distance( key_points[0] ,  key_points[1] )
                pts_dist[i] = float(dist1 + dist2) / float(2)
            elif i ==     len(key_points)-1:
                dist1 = calculate_distance( key_points[len(key_points)-2], key_points[len(key_points)-1] )
                dist2 = calculate_distance( key_points[len(key_points)-1], key_points[0] )
                pts_dist[i] = float(dist1 + dist2) / float(2)                
            else:
                dist1 = calculate_distance( key_points[i], key_points[i - 1] )
                dist2 = calculate_distance( key_points[i], key_points[i + 1] )
                pts_dist[i] = float(dist1 + dist2) / float(2)                
        sort_dict = [(k, pts_dist[k]) for k in sorted(pts_dist, key=pts_dist.get, reverse=False)]    
        idx = sort_dict[0][0]
        sort_dict.clear()
        pts_dist.clear()
        for j in range(len(key_points)):        
            if j != idx:
                aux.append(key_points[j])
        key_points = aux.copy()
        aux.clear()
        j = 0
    #print(len(key_points))
    return key_points

def add_key_points(key_points, points_nr, cnt):
    pts_dist = dict()
    aux = list()
    for x in range(points_nr):
        aux = key_points.copy()
        for i in range(len(key_points)):
            if i == len(key_points)-1:
                dist1 = calculate_distance( key_points[len(key_points)-1], key_points[0] )
                pts_dist[i] = dist1    
            else:
                dist1 = calculate_distance( key_points[i], key_points[i + 1] )
                pts_dist[i] = dist1                
        sort_dict = [(k, pts_dist[k]) for k in sorted(pts_dist, key=pts_dist.get, reverse=True)]
        idx = sort_dict[1][0]        
        ref_point1 = key_points[idx]
        if idx < len(key_points)-1:
            ref_point2 = key_points[idx+1]
        else:
            ref_point2 = key_points[0]
        for y in range(len(cnt)):
            if cnt[y][0][0] == ref_point1[0] and cnt[y][0][1] == ref_point1[1]:            
                if y == len(cnt)-1:
                    cnt_idx = 0
                else:
                    cnt_idx = y + 1
                    
                dist1 = calculate_distance(ref_point1, cnt[cnt_idx][0])
                dist2 = calculate_distance(ref_point2, cnt[cnt_idx][0])
                last_pt_dists = abs(dist2 - dist1)
                
                if cnt_idx == len(cnt)-1:
                    cnt_idx = 0
                else:    
                    cnt_idx = cnt_idx + 1
                    
                dist1 = calculate_distance(ref_point1, cnt[cnt_idx][0])
                dist2 = calculate_distance(ref_point2, cnt[cnt_idx][0])
                act_pt_dists = abs(dist2 - dist1)
                
                while last_pt_dists < act_pt_dists:
                    if cnt_idx == len(cnt)-1:
                        cnt_idx = 0
                    else:    
                        cnt_idx = cnt_idx + 1
                    last_pt_dists = act_pt_dists
                    dist1 = calculate_distance(ref_point1, cnt[cnt_idx][0])
                    dist2 = calculate_distance(ref_point2, cnt[cnt_idx][0])
                    act_pt_dists = abs(dist2 - dist1)
                    
                point_to_add = cnt[cnt_idx][0]
                break                
        key_points.clear()
        for k in range(0, len(aux)+1):
            if k == idx + 1:
                key_points.append(point_to_add)
            elif k <= idx:
                key_points.append(aux[k])
            else:
                key_points.append(aux[k-1])
                
        aux.clear()    
        sort_dict.clear()
        pts_dist.clear()
        #print(len(key_points))
        #print(key_points)
    return key_points    

def hsv_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_range = np.array([0, 60, 80])
    upper_range = np.array([30, 255, 255])

    roi = cv2.inRange(hsv, low_range, upper_range)
    kernel_1 = np.ones( (3, 3), np.uint8)
    kernel_2 = np.ones( (5, 5), np.uint8)
    roi = cv2.erode(roi, kernel_1)
    roi = cv2.dilate(roi, kernel_2)
    return roi    
    
def adjust_key_pts(key_points, cnt):
    if len(key_points) > 20:
        key_points = delete_key_points(key_points, len(key_points) - 20)
    else:
        key_points = add_key_points(key_points, 20 - len(key_points), cnt)
    return key_points    
    
def calculate_avg(data):
    total_sum = 0.0
    for x in data:
        total_sum = total_sum + x    
    avg_value = total_sum / float(len(data))
    return avg_value

def hand_detection_model(img):
    global mypath
    
    frameCopy = img.copy()
    protoFile = "pose_deploy.prototxt"
    weightsFile = "pose_iter_102000.caffemodel"
    
    nPoints = 22
    inHeight = img.shape[0]
    inWidth  = img.shape[1]
    
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    npBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight),(0, 0, 0), swapRB=False, crop=False)
    net.setInput(npBlob)
 
    output = net.forward()
    
    points = []
    #print(output)
    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (inWidth, inHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        print("minVal = "+ str(minVal) + "  prob = " + str(prob) + "  minLoc = " + str(minLoc) + "  point = " + str(point) ) 
        if prob > 0.001 :
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            #cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
     
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)
    cv2.imshow('Output-Keypoints', frameCopy)

def get_line_length_slope2(img, file):
    global features
    global dists
    global slopes
    key_lines = np.zeros( (img.shape[0], img.shape[1]), np.uint8)
    lines_coords = cv2.HoughLinesP(img, 1, np.pi/180,10 , 1000)
    if lines_coords is not None:
        for i in range(0, len(lines_coords)):
            line = lines_coords[i][0]
            dist = math.sqrt( (line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2)
            if dist > 20:
                if line[3] == line[1] or line[2] == line[0] or abs(line[3] - line[1]) < 0.001 or abs(line[2] - line[0]) < 0.001:
                    slope = 0
                else:    
                    slope = (line[3] - line[1]) / (line[2] - line[0])

                dists.append(dist)
                slopes.append(slope)
                cv2.line(key_lines, (line[0], line[1]), (line[2], line[3]), (255,0,0),2)
    features.append(len(dists))
    if len(dists) != 0:
        features.append(calculate_avg(dists))
        features.append(calculate_avg(slopes))
        dists.clear()
        slopes.clear()
    else:
        features.append(0)
        features.append(0)
    
def get_line_length_slope(img, file):
    global features
    key_lines = np.zeros( (img.shape[0], img.shape[1]), np.uint8)
    lines_coords = cv2.HoughLinesP(img, 1, np.pi/180,10 , 1000)
    if lines_coords is not None:
        for i in range(0, len(lines_coords)):
            line = lines_coords[i][0]
            dist = math.sqrt( (line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2)
            if dist > 20:
                if line[3] == line[1] or line[2] == line[0] or abs(line[3] - line[1]) < 0.001 or abs(line[2] - line[0]) < 0.001:
                    slope = 0
                else:    
                    slope = (line[3] - line[1]) / (line[2] - line[0])
                #print ("line = " + str(line))
                #print("slope = " + str(slope))
                features.append(dist)
                features.append(slope)
                cv2.line(key_lines, (line[0], line[1]), (line[2], line[3]), (255,0,0),2)    
    cv2.imshow(file, key_lines)

def skeletonize(img):
    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while(True):
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break    
    skel_letter = cv2.medianBlur(skel, 1)
    kernel = np.ones( (3, 3), np.uint8)
    close_letter = cv2.morphologyEx(skel_letter, cv2.MORPH_CLOSE, kernel)
    close_letter = cv2.dilate(close_letter, kernel)
    
    return close_letter

def count_fingers(pts_list):
    fingers_nr = 0
    global drawing
    finger_found = False
    for i in range(2, len(pts_list)):
        if finger_found is not True:
            start = pts_list[i-2]
            peek = pts_list[i-1]
            end = pts_list[i]
            angle = get_angle(start, end, peek)
            if angle < 90:
                print(angle)
                cv2.circle(drawing,peek,5,[0,0,255],-1)
                fingers_nr = fingers_nr + 1
                finger_found = True
        else:
            finger_found = False
            
    return fingers_nr

def get_angle(start, end, far):
    #determinarea unghiului aplicand teorema lui cosinus
    a = math.sqrt( (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    b = math.sqrt( (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    c = math.sqrt( (end[0] -   far[0]) ** 2 + (end[1] -   far[1]) ** 2)
    angle = math.acos((b ** 2 + c ** 2 - a**2)/ (2 * b * c)    )
    #conversie din radiani in grade
    return ( (angle*180.0)/math.pi )
    
def get_convexity_def(cnt):
    global conv_def
    epsilon = 0.005*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(drawing, [approx], 0, (0,255,0),2)
    hull = cv2.convexHull(cnt)
    cv2.drawContours(drawing, [hull], 0, (255,0,0),2)
    hull = cv2.convexHull(cnt, returnPoints = False)
    defects = cv2.convexityDefects(cnt, hull)
    
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple (cnt[e][0])
        far = tuple (cnt[f][0])
        if d > 3000:
            features.append(d)
    
def get_points(cnt):
    points = list()
    epsilon = 0.005*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    #cv2.drawContours(drawing, [approx], 0, (0,255,0),2)
    hull = cv2.convexHull(cnt)
    #cv2.drawContours(drawing, [hull], 0, (255,0,0),2)
    hull = cv2.convexHull(cnt, returnPoints = False)
    defects = cv2.convexityDefects(cnt, hull)
    
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple (cnt[e][0])
        far = tuple (cnt[f][0])
        if i == 0:
            points.append(start)
        if d > 3000:
            points.append(far)
        points.append(end)    
    return points

def get_all_files(path):
    listOfFile = listdir(path)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(path, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_all_files(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles    

def get_mask(image):
    mask = np.zeros(image.shape, np.uint8)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if x < 17:
                mask[x,y] = 0
            else:
                mask[x,y] = 255            
    return mask

def get_hand_contour(img):  
    cnt = get_max_contour(img)  
    drawing = np.zeros(img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (255, 255, 255), -1)
    return drawing

def adjust_hand_contour(drawing):
    cnt = get_max_contour(drawing)
    x,y,w,h = cv2.boundingRect(cnt)
    symbol = drawing[y : y + h, x : x + w] 
    scale = float(w) / float(h)
    if scale <= 0.8:
        symbol = cv2.resize(symbol, (90, 160))
    elif scale >= 1.2:
        symbol = cv2.resize(symbol, (150, 96))
    else:
        symbol = cv2.resize(symbol, (120, 120))

    symbol = cv2.medianBlur(symbol, 5)
    return symbol
    
def hand_extraction(img):
    cnt = get_max_contour(img)
    drawing = np.zeros(img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (255, 255, 255), -1)
    cnt = get_max_contour(drawing)
    x,y,w,h = cv2.boundingRect(cnt)
    symbol = drawing[y : y+h , x : x + w]
    scale = float(w) / float(h)
    
    if scale <= 0.8:
        symbol = cv2.resize(symbol, (90, 160))
    elif scale >= 1.2:
        symbol = cv2.resize(symbol, (150, 96))
    else:
        symbol = cv2.resize(symbol, (120, 120))
    symbol = cv2.medianBlur(symbol, 5)
    return symbol, cnt
    
def features_calculation(img):
    cnt = get_max_contour(img)
    if len(cnt) >= 20:
        key_points = get_key_points(cnt)
        key_points = adjust_key_pts(key_points, cnt)
        img_with_pts =  np.zeros(img.shape, np.uint8)
        center = compute_center(cnt)
        key_points = create_feature_array(key_points, center)
        return key_points
    else:
        key_points = list()
        key_points.append(0)
        return key_points

def rotate_image(img):
    h, w = img.shape
    center = (h/2, w/2 )
    angle = random.randint(0, 5)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    symbol = cv2.warpAffine( img, rot_matrix, img.shape )
    return symbol

def get_train_data():
    global idx
    global key_points
    global key_points2
    global key_points3 
    mypath ='E:\\Licenta2019\\Sign_Language_translator\\train_dataset'

    onlyfiles = get_all_files(mypath)
    #DETECTIE MANA IN IMAGINE

    train_data = open("train_data.csv", "w")
    train_data.close()

    test_data = open("test_data.csv", "w")
    test_data.close()



    for file in onlyfiles:
        last_slash_pos = file.rfind("\\")
        img_name = file[last_slash_pos+1:len(file)-4]
        print(img_name)
        idx = idx + 1
        splitted_file = file.split("\\")
        label = splitted_file[len(splitted_file)-1][0]
        img = cv2.imread(file, 1)
        height, width, chan = img.shape
        crop_img = img[5:height-15, 5:width-15]    
        hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        thresh1 = cv2.inRange(crop_img, np.array([0,0,0]), np.array([63,80,90]) ) 
        mask=get_mask(thresh1)
        
        thresh1 = cv2.bitwise_and(thresh1, mask)
        kernel1 = np.ones( (5, 5), np.uint8 )
        kernel2 = np.ones( (3, 3), np.uint8 )
        
        thresh1 = cv2.dilate(thresh1, kernel1)
        thresh1 = cv2.erode(thresh1, kernel2)
        thresh1 = cv2.dilate(thresh1, kernel1)
        thresh1 = cv2.erode(thresh1, kernel2)
        thresh1 = cv2.dilate(thresh1, kernel1)
        thresh1 = cv2.erode(thresh1, kernel2)
        
        kernel_1 = np.ones( (3, 3) , np.uint8 )
        kernel_2 = np.ones( (5, 5) , np.uint8 )
        thresh2 = cv2.dilate(thresh1, kernel_1)
        thresh3 = cv2.dilate(thresh1, kernel_2)

        thresh1 = get_hand_contour(thresh1)
        thresh2 = get_hand_contour(thresh2)
        thresh3 = get_hand_contour(thresh3)    
        
        thresh1 = rotate_image(thresh1)
        thresh2 = rotate_image(thresh2)
        thresh3 = rotate_image(thresh3)
      
        
        cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\binary_images2\\'+img_name+'.jpg',thresh1) 
        cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\binary_images2\\'+img_name+'_2.jpg',thresh2) 
        cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\binary_images2\\'+img_name+'_3.jpg',thresh3)       
        
        symbols = list()
        
        symbols.append(adjust_hand_contour(thresh1) )
        symbols.append(adjust_hand_contour(thresh2) )
        symbols.append(adjust_hand_contour(thresh3) )
        
        
        img_with_pts = np.zeros(symbols[0].shape, np.uint8)
        
        
        for i in range(len(symbols)):
            cnt = get_max_contour(symbols[i])
            img_with_pts = np.zeros(symbols[0].shape, np.uint8)
            if len(cnt) > 20:
                key_points = get_key_points(cnt)
                key_points = adjust_key_pts(key_points, cnt) 
                key_points_length = len(key_points) 
                center = compute_center(cnt)
                for x in range(len(key_points)):
                    cv2.circle(img_with_pts, tuple(key_points[x]), 1, (255,255,255))
                if i == 0:    
                    cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\black_white_images\\'+img_name+'.jpg',img_with_pts)
                    cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\dilated_binary_images2\\'+img_name+'.jpg',symbols[i])
                elif i == 1:
                    cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\black_white_images\\'+img_name+'a2.jpg',img_with_pts)
                    cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\dilated_binary_images2\\'+img_name+'a2.jpg',symbols[i])
                elif i == 2:
                    cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\black_white_images\\'+img_name+'a3.jpg',img_with_pts)
                    cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\dilated_binary_images2\\'+img_name+'a3.jpg',symbols[i]) 
                key_points = create_feature_array(key_points, center)
                write_to_csv(label, key_points)
        
def delete_noize(frames):
    result = cv2.bitwise_and(frames[0], frames[1])
    for i in range(2, len(frames)):
        result = cv2.bitwise_and(result, frames[i])
    return result

#get_train_data()
    
#cv2.waitKey(0)

