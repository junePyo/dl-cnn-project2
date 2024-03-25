import cv2
import os
import glob

img_dir = 'face_images'
files = glob.glob(os.path.join(img_dir, '*.jpg'))

image_ = cv2.imread(files[0])
imageLAB = cv2.cvtColor(image_, cv2.COLOR_BGR2LAB)
l_, a_, b_=cv2.split(imageLAB)

cv2.imshow('image', image_)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('imageLAB', imageLAB)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('L',l_)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('a',a_)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('b',b_)
cv2.waitKey(0)
cv2.destroyAllWindows()
