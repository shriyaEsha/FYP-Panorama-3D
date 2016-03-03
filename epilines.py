import cv2
import numpy as np
import random
from matplotlib import pyplot as plt


def drawlines(img1,img2,lines,pts1,pts2,fname):
    ''' img1 - image on which we draw the epilines for the points in img1
        lines - corresponding epilines ''' 
    f = open(fname,"w")
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    (mask1,mask2,frame1,frame2) = (img1,img2,img1,img2)
    for (r,pt1,pt2) in zip(lines,pts1,pts2):
	#print r,pt1,pt2
	color = tuple(np.random.randint(0,255,3).tolist())
	#print 'color: ',color
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
	l = "("+str(x0)+","+str(y0)+")-->"+"("+str(x1)+","+str(y1)+")\n";
	f.write(l)	
	cv2.line(mask1, (x0,y0), (x1,y1), color,1)
	cv2.circle(frame1,tuple(pt1),5,color,-1)
	img1 = cv2.add(frame1,mask1)
	cv2.circle(mask2,tuple(pt2),5,color,-1)
	img2 = cv2.add(img2,mask2)
    f.close()
    return img1,img2
def draw_correspondences(image1, image2, points1, points2):
  'Connects corresponding features in the two images using yellow lines.'

  ## Put images side-by-side into 'image'.
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  image = np.zeros((max(h1, h2), w1 + w2), np.uint8)
  image[:h1, :w1] = image1
  image[:h2, w1:w1+w2] = image2
  
  ## Draw yellow lines connecting corresponding features.
  for (x1, y1), (x2, y2) in zip(np.int32(points1), np.int32(points2)):
    cv2.line(image, (x1, y1), (x2+w1, y2), (2555, 0, 255), lineType=cv2.CV_AA)
  return image

img1 = cv2.imread('aloeL.jpg',0)   # queryimage # left image
img2 = cv2.imread('aloeR.jpg',0) #trainimage    # right image
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        
pts2 = np.float32(pts2)
pts1 = np.float32(pts1)       
im = draw_correspondences(img1,img2,pts1,pts2)
cv2.imwrite("correspondences.jpg", im)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
# We select only first 100 points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]
random.shuffle(pts1)
random.shuffle(pts2)
pts1 = pts1[:100]
pts2 = pts2[:100]
# drawing lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2,"img1.txt")
# drawing lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2),1,F)
lines2 = lines2.reshape(-1,3)

img3,img4 = drawlines(img2,img1,lines2,pts2,pts1,"img2.txt")
cv2.imwrite('left6.13.jpg',img5)
cv2.imwrite('right6.13.jpg',img3)

