import Image
from numpy import *

from scipy.ndimage import geometric_transform
from scipy.misc import imsave

def polar_warp(im,sz,r):
 """ Warp an image to a square polar version. """
 
 h,w = im.shape[:2]
  
 def polar_2_rect(xy):
  """ Convert polar coordinates to coordinates in 
   the original image for geometric_transform. """
   
  x,y = float(xy[1]),float(xy[0])
  theta = 0.5*arctan2(x-sz/2,y-sz/2)
  R = sqrt( (x-sz/2)*(x-sz/2) + (y-sz/2)*(y-sz/2) ) - r
  
  return (2*h*R/(sz-2*r),w/2+theta*w/pi+pi/2)
 
 # if color image, warp each channel, otherwise there's just one
 if len(im.shape)==3: 
  warped = zeros((sz,sz,3))
  for i in range(3):
   warped[:,:,i] = geometric_transform(im[:,:,i],polar_2_rect,output_shape=(sz,sz),mode='nearest')
 else:
  warped = geometric_transform(im,polar_2_rect,output_shape=(sz,sz),mode='nearest')
 
 return warped


if __name__ == "__main__":
 
 # load image from http://www.filipecalegario.com/comppho/3assignment/
 im = array(Image.open('autostitch-stitch.jpg'))
 h,w = im.shape[:2]
 
 # flip vertical to make math easier for warp
 im = flipud(im)
 
 # warp image
 wim = uint8(polar_warp(im,h,5))
 
 # save
 imsave('polar.png',wim)
