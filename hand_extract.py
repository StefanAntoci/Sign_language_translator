import cv2
import numpy as np

img = cv2.imread('E:\\Licenta2019\\sign-language-alphabet-recognizer-master\\dataset\\A\\A1.jpg', 1)

height, width, chan = img.shape
crop_img = img[5:height-5, 5:width-5]

blue, green, red = cv2.split(crop_img)

blur = cv2.GaussianBlur(red, (5, 5), 0)	

ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)



_, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

drawing = np.zeros(img.shape, np.uint8)

max_area = 0.0
ci = 0

print (len(contours))
for i in range (len(contours)):
	cnt = contours[i]
	area = cv2.contourArea(cnt)
	if (area > max_area):
		max_area = area
		ci = i		


		
cnt = contours[ci]
hull = cv2.convexHull(cnt)
moments = cv2.moments(cnt)

if moments['m00']!=0:
	cx = int(moments['m10']/moments['m00'])
	cy = int(moments['m01']/moments['m00'])

centr = (cx, cy)
cv2.circle(red, centr, 5, [0,0,255],2)
cv2.drawContours(drawing, [cnt], 0, (0,255,0),2)
cv2.drawContours(drawing, [hull], 0, (0,0,255),2)

cnt = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
hull = cv2.convexHull(cnt,returnPoints = False)

if(1):
	defects = cv2.convexityDefects(cnt, hull)
	print(defects)
	mind = 0
	maxd = 0
	for i in range(defects.shape[0]):
		s,e,f,d = defects[i,0]
		start = tuple(cnt[s][0])
		end = tuple(cnt[e][0])
		far = tuple(cnt[f][0])
		dist = cv2.pointPolygonTest(cnt, centr, True)
		cv2.line(crop_img, start, end, [0,0,255],2)
		
		cv2.circle(crop_img, far, 5, [0,0,255],-1)
	
	print(i)

cv2.imshow('output',drawing)
cv2.imshow('input', img)
cv2.waitKey(0)

