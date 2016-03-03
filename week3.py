# coding: utf-8
import cv2, numpy as np
from find_obj import init_feature, filter_matches, explore_match
## 1. Extract SURF keypoints and descriptors from an image. [4] ----------
def extract_features(image, surfThreshold=5000):

  ## TODO: Convert image to grayscale (for SURF detector).
  ## TODO: Detect SURF features and compute descriptors.
  ## TODO: (Overwrite the following 2 lines with your answer.)
  g_i = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  surf = cv2.SURF(surfThreshold)
  kp, des = surf.detectAndCompute(g_i,None)
  print len(des)
  return (kp, des)


## 2. Find corresponding features between the images. [2] ----------------
def find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2):

  ## Find corresponding features.
  raw_matches = matcher.knnMatch(descriptors1, trainDescriptors = descriptors2, k = 2)
  points1, points2, kp_pairs = filter_matches(keypoints1, keypoints2, raw_matches)
  print points1,points2
  ## TODO: Look up corresponding keypoints.
  ## TODO: (Overwrite the following 2 lines with your answer.)
  
  return (points1, points2)


## 3. Calculate the size and offset of the stitched panorama. [5] --------
def calculate_size(size_image1, size_image2, homography):
  
  ## TODO: Calculate the size and offset of the stitched panorama.
  ## TODO: (Overwrite the following 2 lines with your answer.)
  offset = (0, 0)
  #1600 X 600
  size   = (1600, 600)
  
  ## Update the homography to shift by the offset - X and Y translation by offset
  homography[0:2,2] +=  offset

  return (size, offset)


## 4. Combine images into a panorama. [4] --------------------------------
def merge_images(image1, image2, homography, size, offset, keypoints):

  ## TODO: Combine the two images into one.
  ## TODO: (Overwrite the following 5 lines with your answer.)
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  print h1,w1,h2,w2,size[0],size[1]
  size = (w1+w2+1,h1+h2)
  panorama = np.zeros((size[1], size[0], 3), np.uint8)
  #set the 2 images side by side
  panorama[:h1, :w1] = image1
  panorama[:h2, w1:w1+w2] = image2
  (ox,oy) = offset
  translation = np.matrix([
    [1.0, 0.0, ox],
    [0, 1.0, oy],
    [0.0, 0.0, 1.0]
  ])
  homography = translation * homography
  cv2.warpPerspective(image2, homography, size, panorama)
  panorama[oy:h1+oy, ox:ox+w1] = image1  
  ## TODO: Draw the common feature keypoints.

  return panorama


##---- No need to change anything below this point. ----------------------


def match_flann(desc1, desc2, r_threshold = 0.06):
  'Finds strong corresponding features in the two given vectors.'
  ## Adapted from <http://stackoverflow.com/a/8311498/72470>.

  ## Build a kd-tree from the second feature vector.
  FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
  flann = cv2.flann_Index(desc2, {'algorithm': FLANN_INDEX_KDTREE, 'trees': 4})

  ## For each feature in desc1, find the two closest ones in desc2.
  (idx2, dist) = flann.knnSearch(desc1, 2, params={}) # bug: need empty {}

  ## Create a mask that indicates if the first-found item is sufficiently
  ## closer than the second-found, to check if the match is robust.
  mask = dist[:,0] / dist[:,1] < r_threshold
  
  ## Only return robust feature pairs.
  idx1  = np.arange(len(desc1))
  pairs = np.int32(zip(idx1, idx2[:,0]))
  return pairs[mask]


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
    cv2.line(image, (x1, y1), (x2+w1, y2), (0, 255, 255), lineType=cv2.CV_AA)

  return image


if __name__ == "__main__":

  ## Load images.
  image1 = cv2.imread("p1.jpg")
  image2 = cv2.imread("p2.jpg")
  import sys, getopt
  opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
  opts = dict(opts)
  feature_name = opts.get('--feature', 'surf-flann')
  detector, matcher = init_feature(feature_name)
  
  ## Detect features and compute descriptors.
  (keypoints1, descriptors1) = extract_features(image1)
  (keypoints2, descriptors2) = extract_features(image2)
  print len(keypoints1), "features detected in image1"
  print len(keypoints2), "features detected in image2"
  
  ## Find corresponding features.
  (points1, points2) = find_correspondences(keypoints1, descriptors1, keypoints2, descriptors2)
  print len(points1), "features matched"

  ## Visualise corresponding features.
  correspondences = draw_correspondences(image1, image2, points1, points2)
  cv2.imwrite("correspondences.jpg", correspondences)
  cv2.imshow('correspondences', correspondences)
  
  ## Find homography between the views.
  (homography, _) = cv2.findHomography(points2, points1)
  print homography
  ## Calculate size and offset of merged panorama.
  (size, offset) = calculate_size(image1.shape, image2.shape, homography)
  print "output size: %ix%i" % size
  
  ## Finally combine images into a panorama.
  panorama = merge_images(image1, image2, homography, size, offset, (points1, points2))
  cv2.imwrite("panorama.jpg", panorama)
  cv2.imshow('panorama', panorama)
  
  ## Show images and wait for escape key.
  while cv2.waitKey(100) != 27:
    pass
