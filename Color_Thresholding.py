import glob
import cv2
import numpy as np

filenames = [file for file in glob.glob("E:\\MS\\Research\\USL\\Code\\Object_detection\\Ground_Test_Images\\*.jpg")]

for image in filenames:
       
       ## Test_1
       #org = cv2.imread('Test1.jpg')
       #lower_red = np.array([60, 50, 0])
       #upper_red = np.array([200,150, 200])
       
       # Test_2
       # Ground Images
       org = cv2.imread(image)
       lower_red = np.array([0, 100, 0])
       upper_red = np.array([18,255, 255])
       
       # Test 4
       #org = cv2.imread('Test4.jpg')
       #lower_red = np.array([0, 0, 140])
       #upper_red = np.array([180,72, 255])
       
       ## Test 5-10
       #org = cv2.imread(image)
       ## For aerial images from DJI
       #lower_red = np.array([0, 0, 150])
       #upper_red = np.array([15,150, 255])
       
       hsv = cv2.cvtColor(org, cv2.COLOR_BGR2HSV)
       
       brickMask = cv2.inRange(hsv, lower_red, upper_red)
       res = cv2.bitwise_and(org, org, mask = brickMask)
       
       kernel = np.ones((5,5), np.uint8)
       erosion = cv2.erode(brickMask, kernel,iterations = 1)
       
       opening = cv2.morphologyEx(brickMask, cv2.MORPH_OPEN, kernel)
       closing = cv2.morphologyEx(brickMask, cv2.MORPH_CLOSE, kernel)
       
       # Removes false positive
       # More reductive - could miss out on an actual brick
       resOpen = cv2.bitwise_and(org, org, mask = opening)
       
       # Removes false negatives
       # More inclusive - could misidentify some other things as bricks
       #                  safer approach
       resClose = cv2.bitwise_and(org, org, mask = closing)
       
       #cv2.imshow('Original',org)
       #cv2.imshow('Mask', brickMask)
       #cv2.imshow('Result',res)
       #cv2.imshow('resOpen', resOpen)
       #cv2.imshow('resClose', resClose)
       
       #cv2.imshow('Canny', cv2.Canny(org, 300, 300))
       imagenoext = image[:-4]
       cv2.imwrite(imagenoext+"_colorthresholded.jpg",resClose)
#"E:\\MS\\Research\\USL\\Code\\Object_detection\\Aerial_Test_Images\\ColorThresholded\\"+
cv2.waitKey(0)
cv2.destroyAllWindows()


