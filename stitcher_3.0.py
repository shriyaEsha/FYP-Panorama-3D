# import the necessary packages
import Stitcher
import argparse
import imutils
import cv2
import time

def pano(images,i):
  if (i+1) <= (len(images)-1):
    try:
      image1 = cv2.imread(images[i])
      image1 = imutils.resize(image1, width=400)
    except TypeError:
      image1 = images[i]
    try:
      image2 = cv2.imread(images[i+1])
      image2 = imutils.resize(image2, width=400)
    except TypeError:
      image2 = images[i+1]

  else:
  # print 'OOB'
  return  
  # stitch the images together to create a panorama
  stitcher = Stitcher.Stitcher()
  (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
	 
  # show the images
  cv2.imwrite("yosemite/corr.jpg", vis)

  if(len(images) == 2):#final panorama
    filename = "yosemite/pano_ multi_final"+str(i)+".jpg"
    print 'pano size: ',images[0].shape[:2]
  else:
    filename = "yosemite/pano_multi"+str(i)+".jpg"
  cv2.imwrite(filename,images[i])


if __name__ == "__main__":
	st = time.time()
	images = ["yosemite/yosemite1.jpg","yosemite/yosemite2.jpg","yosemite/yosemite3.jpg","yosemite/yosemite4.jpg"]
	n = len(images)
	val = int(math.ceil(math.log(len(images),2)))+1
	for q in range(val):
	    for i in xrange(0,len(images),1):
	          print 'i: ',i,' len: ',len(images)
	          t = Thread(target=pano, args=(images,i))
	          t.start()
	          t.join()
	          if i+1 <= len(images)-1:
	            del images[i+1]
	et = time.time()
	print 'time: ',et-st,'s' 

