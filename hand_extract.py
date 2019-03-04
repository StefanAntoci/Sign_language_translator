import cv2
import numpy as np
import math
from os import listdir
from os.path import isfile, join

#Trasaturile din imagine de care avem nevoie
nr_convex_def = 0 #numarul defectelor de convexitate din interiorul mainii
convex_def_depth = list() #lista cu adancimea defectelor de convexitate din interiorul mainii 
nr_fingers = 0 # numarul degetelor

#def remove_lines(img, height, width)

def get_angle(start, end, far):
	#determinarea unghiului aplicand teorema lui cosinus
	a = math.sqrt( (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
	b = math.sqrt( (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
	c = math.sqrt( (end[0] -   far[0]) ** 2 + (end[1] -   far[1]) ** 2)
	angle = math.acos((b ** 2 + c ** 2 - a**2)/ (2 * b * c)	)
	#conversie din radiani in grade
	return ( (angle*180.0)/math.pi )
	
	
mypath = 'E:\\Licenta2019\\Sign_Language_translator\\dataset2'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#backsubmog = cv2.createBackgroundSubtractorMOG2()
# DETECTIE MANA IN IMAGINE



for file in onlyfiles:
	#print(file)
	img = cv2.imread(mypath + '\\' + file, 1)
	flipped = cv2.flip(img, 1)
	height, width, chan = img.shape
	#print(str(height) + "  " + str(width) )
	#marginile imaginii nu permit detectarea corecta a mainii, de asta facem crop la imagine
	crop_img = img[5:height-15, 5:width-15]
	#thresh1 = backsubmog.apply(gray)
	
	blue, green, red = cv2.split(img)

	gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

	blur = cv2.GaussianBlur(gray,(3,3),0)
	ret,thresh1 = cv2.threshold(blue,70,255,cv2.THRESH_BINARY_INV)

	_, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
	cv2.drawContours(drawing, [cnt], 0, (0,255,0),2)
	#cnt = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
	hull = cv2.convexHull(cnt, returnPoints = False)
	defects = cv2.convexityDefects(cnt, hull)
	
	print(file)
	for i in range(defects.shape[0]):
		s,e,f,d = defects[i,0]
		start = tuple(cnt[s][0])
		end = tuple (cnt[e][0])
		far = tuple (cnt[f][0])
		cv2.line(drawing, start, end, [255,0,0],2)
		cv2.circle(drawing,far,5,[0,0,255],-1)
		angle = get_angle(start, end, far)
		print('distance = ' + str(d))
		
	print('Numarul defectelor de convexitate = {}'.format(nr_convex_def))
	print('Adancimea defectelor de convexitate')
	print(convex_def_depth)
	print('Numarul degetelor = {}'.format(nr_fingers))
	
	nr_fingers = 0
	nr_convex_def = 0
	convex_def_depth.clear()
	cv2.imshow(file + 'contours', drawing)
	

cv2.waitKey(0)

