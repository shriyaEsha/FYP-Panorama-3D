import cv

def cut(disparity, image, threshold):
 print 'size: ',image.width,image.height
 raw_input()
 for i in range(0, image.height):
  for j in range(0, image.width):
   # keep closer object
   if cv.GetReal2D(disparity,i,j) > threshold:
    cv.Set2D(disparity,i,j,cv.Get2D(image,i,j))

# loading the stereo pair
left  = cv.LoadImage('LR/left3.jpg',cv.CV_LOAD_IMAGE_GRAYSCALE)
right = cv.LoadImage('LR/right3.jpg',cv.CV_LOAD_IMAGE_GRAYSCALE)

disparity_left  = cv.CreateMat(left.height, left.width, cv.CV_16S)
disparity_right = cv.CreateMat(left.height, left.width, cv.CV_16S)

print 'here'
raw_input()
# data structure initialization
state = cv.CreateStereoGCState(16,2)
print 'here'
raw_input()
# running the graph-cut algorithm
cv.(left,right,disparity_left,disparity_right,state)
print 'here'
raw_input()
disp_left_visual = cv.CreateMat(left.height, left.width, cv.CV_8U)
cv.ConvertScale( disparity_left, disp_left_visual, -16 );
cv.Save( "disparity.pgm", disp_left_visual ); # save the map
print 'here'
raw_input()
# cutting the object farthest of a threshold (120)
cut(disp_left_visual,left,120)

cv.NamedWindow('Disparity map', cv.CV_WINDOW_AUTOSIZE)
cv.ShowImage('Disparity map', disp_left_visual)
cv.WaitKey()