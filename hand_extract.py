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

def write_to_csv(label, key_points, areas):    
    global idx
    print(idx)
    if idx % 50 == 0:
        test_data = open("test_data.csv", "a")
        for data in key_points:
            test_data.write(str(data) )
            test_data.write(" , ")
        
        for area in areas:
            test_data.write(str(area) )
            test_data.write(" , ")
        test_data.write(label)
        test_data.write("\n")
        test_data.close()
        key_points.clear()
        
       
    else:
        train_data = open("train_data.csv", "a")
        for data in key_points:
            train_data.write(str(data) )
            train_data.write(" , ")

        for area in areas:
            train_data.write(str(area) )
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
    
    if len(contours) == 0:
        cnt = list()
        return cnt    
    drawing = np.zeros(img.shape, np.uint8)
    max_area = 0.0
    ci = 0
    for i in range (len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area > max_area):
            max_area = area
            ci = i        
    
    cnt = contours[ci]
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
        while dist < 5:
            if i >= pts_number:
                break
            end_point = cnt[i][0]
            dist = calculate_distance(start_point, end_point)
            i = i + 2
        if dist >= 5:
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
    if len(key_points) > 40:
        key_points = delete_key_points(key_points, len(key_points) - 40)
    else:
        key_points = add_key_points(key_points, 40 - len(key_points), cnt)
    return key_points    
    
def calculate_avg(data):
    total_sum = 0.0
    for x in data:
        total_sum = total_sum + x    
    avg_value = total_sum / float(len(data))
    return avg_value

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
    if len(cnt) != 0:
        x,y,w,h = cv2.boundingRect(cnt)
        symbol = drawing[y : y + h, x : x + w] 
        scale = float(w) / float(h)
        round(scale, 2)
        height = math.sqrt(float(14400) / float(scale) )
        height = int(height)
        width = 14400/height
        width = int(width)    
        symbol = cv2.resize( symbol, (width, height) )
        symbol = cv2.medianBlur(symbol, 5)    
        return symbol   
    else:
        return list()   

def get_4areas(symbol):
    
    
    try:
        section_areas = list()
        cnt = get_max_contour(symbol)
        all_area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        
        left_down = symbol[y : int(h/2), x : int(w/2) ]
        left_down_cnt = get_max_contour(left_down)
        left_down_area = cv2.contourArea(left_down_cnt)
        section_areas.append( float(left_down_area) / float(all_area) ) 
        
        right_down = symbol[y : int(h/2), int(w/2) : w]
        right_down_cnt = get_max_contour(right_down)
        right_down_area = cv2.contourArea(right_down_cnt)
        section_areas.append( float(right_down_area) / float(all_area) ) 

        left_up = symbol[int(h/2) : h, x : int(w/2)]
        left_up_cnt = get_max_contour(left_up)
        left_up_area = cv2.contourArea(left_up_cnt)
        section_areas.append( float(left_up_area) / float(all_area) ) 

        right_up = symbol[int(h/2) : h, int(w/2) : w]
        right_up_cnt = get_max_contour(right_up)
        right_up_area = cv2.contourArea(right_up_cnt)
        section_areas.append( float(right_up_area) / float(all_area) )
    except TypeError:
        section_areas = list()
        section_areas.append(0)
        
    return section_areas
    
def features_calculation(img):
    cnt = get_max_contour(img)
    if len(cnt) >= 40:
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
    mypath ='E:\\Licenta2019\\Sign_Language_translator\\video_records3\\MoreData'

    onlyfiles = get_all_files(mypath)
    #DETECTIE MANA IN IMAGINE

    train_data = open("train_data.csv", "a")
    train_data.close()

    test_data = open("test_data.csv", "a")
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
        
        thresh1 = cv2.inRange(crop_img, np.array([10,10,10]), np.array([255,255,255]) ) 
        #mask=get_mask(thresh1)
        
        #thresh1 = cv2.bitwise_and(thresh1, mask)
        kernel1 = np.ones( (5, 5), np.uint8 )
        kernel2 = np.ones( (3, 3), np.uint8 )
        
        thresh1 = cv2.dilate(thresh1, kernel1)
        thresh1 = cv2.erode(thresh1, kernel2)
        thresh1 = cv2.dilate(thresh1, kernel1)
        thresh1 = cv2.erode(thresh1, kernel2)
        thresh1 = cv2.dilate(thresh1, kernel1)
        thresh1 = cv2.erode(thresh1, kernel2)
        
        kernel_1 = np.ones( (3, 3) , np.uint8 )
        #kernel_2 = np.ones( (5, 5) , np.uint8 )
        #thresh2 = cv2.dilate(thresh1, kernel_1)
        #thresh3 = cv2.dilate(thresh1, kernel_2)

        thresh1 = get_hand_contour(thresh1)
        # thresh2 = get_hand_contour(thresh2)
        # thresh3 = get_hand_contour(thresh3)    
        
        # thresh1 = rotate_image(thresh1)
        # thresh2 = rotate_image(thresh2)
        # thresh3 = rotate_image(thresh3)
      
        
        #cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\myRotatedImgs\\'+img_name+'.jpg',thresh1) 
        #cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\myRotatedImgs\\'+img_name+'_2.jpg',thresh2) 
        #cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\myRotatedImgs\\'+img_name+'_3.jpg',thresh3)       
        
        symbols = list()
        
        symbols.append(adjust_hand_contour(thresh1) )
        cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\video_records3\\symbols_cris\\'+img_name+'.jpg',symbols[0])
        # symbols.append(adjust_hand_contour(thresh2) )
        # symbols.append(adjust_hand_contour(thresh3) )
        # cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\myAdjusted\\'+img_name+'.jpg',symbols[1])
        # cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\myAdjusted\\'+img_name+'.jpg',symbols[2])        
        
        img_with_pts = np.zeros(symbols[0].shape, np.uint8)
        
        
        for i in range(len(symbols)):
            cnt = get_max_contour(symbols[i])
            img_with_pts = np.zeros(symbols[0].shape, np.uint8)
            if len(cnt) > 40:
                key_points = get_key_points(cnt)
                key_points = adjust_key_pts(key_points, cnt) 
                key_points_length = len(key_points)    
                center = compute_center(cnt)
                for x in range(len(key_points)):
                    cv2.circle(img_with_pts, tuple(key_points[x]), 1, (255,255,255))  
                cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\video_records3\\points_cris\\'+img_name+'.jpg',img_with_pts)        
                # if i == 0:    
                    # cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\myPoints\\'+img_name+'.jpg',img_with_pts)
                # elif i == 1:
                    # cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\myPoints\\'+img_name+'a2.jpg',img_with_pts)
                # elif i == 2:
                    # cv2.imwrite('E:\\Licenta2019\\Sign_Language_translator\\myPoints\\'+img_name+'a3.jpg',img_with_pts)
                key_points = create_feature_array(key_points, center)
                areas = get_4areas(symbols[i])
                write_to_csv(label, key_points, areas)
        
def delete_noize(frames):
    result = cv2.bitwise_and(frames[0], frames[1])
    for i in range(2, len(frames)):
        result = cv2.bitwise_and(result, frames[i])
    return result

def determine_scale():
    global idx
    mypath ='E:\\Licenta2019\\Sign_Language_translator\\myDataset\\D2'
    onlyfiles = get_all_files(mypath)        

    train_data = open("train_data.csv", "w")
    train_data.close()

    test_data = open("test_data.csv", "w")
    test_data.close()
    
    for file in onlyfiles:
        idx  = idx + 1
        print(idx)
        last_slash_pos = file.rfind("\\")
        img_name = file[last_slash_pos+1:len(file)-4]
        print(img_name)
        splitted_file = file.split("\\")
        label = splitted_file[len(splitted_file)-1][0]
        img = cv2.imread(file, 1)
        height, width, chan = img.shape

        thresh1 = cv2.inRange(img, np.array([10,10,10]), np.array([255,255,255]) ) 
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
    
        thresh1 = get_hand_contour(thresh1)
        symbol = adjust_hand_contour(thresh1)
        img_with_pts = np.zeros(symbol.shape, np.uint8)
        cnt = get_max_contour(symbol)
        areas = get_4areas(symbol)
        print(cnt)
        if len(cnt) > 40:
            key_points = get_key_points(cnt)
            key_points = adjust_key_pts(key_points, cnt)  
            center = compute_center(cnt)    
            key_points = create_feature_array(key_points, center)    
            write_to_csv(label, key_points, areas)
#determine_scale()                
#get_train_data()
    
#cv2.waitKey(0)
