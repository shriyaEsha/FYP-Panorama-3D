import cv2, numpy as np
import math
import argparse as ap
import time
from threading import Thread

#Extract SURF keypoints and descriptors from an image
def extract_features(image1,image2, surfThreshold=1000, algorithm='SURF'):
  # Convert image to grayscale (for SURF detector).
  try:
    # print 'Type of image for cvtColor: ',type(image1)
    image_gs1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
  except TypeError:
    return
  try:
    # print 'Type of image for cvtColor: ',type(image2)
    image_gs2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
  except TypeError:
    return

  
  # Detect SURF features and compute descriptors.
  detector = cv2.xfeatures2d.SURF_create()
  (keypoints1,descriptors1) = detector.detectAndCompute(image_gs1,None)
  (keypoints2,descriptors2) = detector.detectAndCompute(image_gs2,None)
  
  return (keypoints1, descriptors1,keypoints2, descriptors2)


# Find corresponding features between the images
def find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2):

  # Find corresponding features.
  match = match_flann(descriptors1, descriptors2)
  
  points1 = np.array([keypoints1[i].pt for (i, j) in match], np.float32)
  points2 = np.array([keypoints2[j].pt for (i, j) in match], np.float32)
  
  return (points1, points2)


#Calculate the size and offset of the stitched panorama
def calculate_size(size_image1, size_image2, homography):
  
  (h1, w1) = size_image1[:2]
  (h2, w2) = size_image2[:2]
  
  #remap the coordinates of the projected image onto the panorama image space
  top_left = np.dot(homography,np.asarray([0,0,1]))
  top_right = np.dot(homography,np.asarray([w2,0,1]))
  bottom_left = np.dot(homography,np.asarray([0,h2,1]))
  bottom_right = np.dot(homography,np.asarray([w2,h2,1]))

  #normalize
  top_left = top_left/top_left[2]
  top_right = top_right/top_right[2]
  bottom_left = bottom_left/bottom_left[2]
  bottom_right = bottom_right/bottom_right[2]

  pano_left = int(min(top_left[0], bottom_left[0], 0))
  pano_right = int(max(top_right[0], bottom_right[0], w1))
  W = pano_right - pano_left
  
  pano_top = int(min(top_left[1], top_right[1], 0))
  pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
  H = pano_bottom - pano_top
  
  size = (W, H)
  
  # offset of first image relative to panorama
  X = int(min(top_left[0], bottom_left[0], 0))
  Y = int(min(top_left[1], top_right[1], 0))
  offset = (-X, -Y)
  return (size, offset)

## 4. Combine images into a panorama. 
def merge_images(image1, image2, homography, size, offset, keypoints):

  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  
  panorama = np.zeros((size[1], size[0], 3), np.uint8)
  
  (ox, oy) = offset
  
  translation = np.matrix([
    [1.0, 0.0, ox],
    [0, 1.0, oy],
    [0.0, 0.0, 1.0]
  ])
  
  
  # draw the transformed image2
  cv2.warpPerspective(image2, homography, size, panorama)
  cv2.imshow("pano",panorama)
  cv2.waitKey(0)
  panorama[oy:h1+oy, ox:ox+w1] = image1  
 
  return panorama


def match_flann(des1, des2, r_threshold = 0.12):
  FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks=50)   # or pass empty dictionary

  flann = cv2.FlannBasedMatcher(index_params,search_params)

  matches = flann.knnMatch(des1,des2,k=2) #returns 2 closest matches
  good = []
  for m,n in matches:
    if m.distance < 0.75 * n.distance:
      good.append([m])

  matchesMask = [[0,0] for i in xrange(len(matches))]
  print 'good matches '
  raw_input()
  print good
  raw_input()
# ratio test as per Lowe's paper
  for i,(m,n) in enumerate(matches):
      if m.distance < 0.7*n.distance:
          matchesMask[i]=[1,0] #take first descriptor

  draw_params = dict(matchColor = (0,255,0),
                     singlePointColor = (255,0,0),
                     matchesMask = matchesMask,
                     flags = 0)

  # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None)

  # plt.imshow(img3,),plt.show()
  #write kp and match indexes to files
  m = []
  idx1 = np.arange(len(des1))
  idx2 = []
  for a in good:
    if a[0].trainIdx > min(len(des1)-1,len(des2)-1):
      print 'stoppp ',a[0].trainIdx
    else:
      idx2.append(a[0].trainIdx)
  print idx2
  raw_input()
  m = np.int32(zip(idx1,idx2))
  return m

  
def draw_correspondences(image1, image2, points1, points2):
  'Connects corresponding features in the two images using yellow lines.'

  ## Put images side-by-side into 'image'.
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  image = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
  image[:h1, :w1] = image1
  image[:h2, w1:w1+w2] = image2
  
  ## Draw yellow lines connecting corresponding features.
  for (x1, y1), (x2, y2) in zip(np.int32(points1), np.int32(points2)):
    cv2.line(image, (x1, y1), (x2+w1, y2), (255, 0, 255), lineType=cv2.LINE_AA)

  return image
def pano(image1,image2,errVal):
    ## Detect features and compute descriptors.
  (keypoints1, descriptors1,keypoints2, descriptors2) = extract_features(image1,image2)
  print len(keypoints1), "features detected in image1"
  print len(keypoints2), "features detected in image2"
  
  ## Find corresponding features.
  (points1, points2) = find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2)
  print len(points1), "features matched"
  
  ## Visualise corresponding features.
  correspondences = draw_correspondences(image1, image2, points1, points2)
  cv2.imwrite("yosemite/correspondences_orig.jpg", correspondences)
  print 'Wrote correspondences.jpg'
  
  try:
  ## Find homography between the views.
    (homography, _) = cv2.findHomography(points2, points1)
  except Exception:
    print 'Not enough matches!'
    errVal = 0
    return 
  ## Calculate size and offset of merged panorama.
  (size, offset) = calculate_size(image1.shape, image2.shape, homography)
  print "output size: %ix%i" % size
  
  ## Finally combine images into a panorama.
  panorama = merge_images(image1, image2, homography, size, offset, (points1, points2))
  errVal = 1
  return panorama

if __name__ == "__main__":
  
  # images = ["house/h1.jpg","house/h2.jpg","house/h3.jpg"]
  # images = ["hostel/01.jpg","hostel/02.jpg","hostel/03.jpg","hostel/04.jpg"]
  # images = ["hostel/1.jpg","hostel/2.jpg","hostel/3.jpg","hostel/4.jpg"]
  images = ["yosemite/yosemite1.jpg","yosemite/yosemite2.jpg","yosemite/yosemite3.jpg","yosemite/yosemite4.jpg"]
  im1 = cv2.imread(images[0])
  im2 = cv2.imread(images[1])
  errVal = 1
  p = 2
  panorama = pano(im1,im2,errVal)
  if not errVal:
    panorama = im1
  cv2.imwrite("yosemite/panorama24.jpg", panorama)
  images[:] = images[2:]
  length = len(images)#3
  for i in xrange(length):
    errVal = 1
    print 'length: ',length
    print 'images : ',i," ",images
    im1 = panorama
    im2 = cv2.imread(images[0])
    panorama = pano(im1,im2,errVal)
    images[:] = images[1:]  
    cv2.imwrite("yosemite/panorama"+str(p)+".jpg", panorama)
    p += 1
  print 'Wrote panorama.jpg'