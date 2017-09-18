import cv2
from PIL import Image
import numpy as np
import time

# thug life meme mask image path
maskPath = "mask.png"
# haarcascade path
cascPath = "haarcascade_frontalface_default.xml"

# cascade classifier object 
faceCascade = cv2.CascadeClassifier(cascPath)

# Open mask as PIL image
mask = Image.open(maskPath)

def thug_mask(image):
	"""
	function to add thug life mask to input image
	"""

	# convert input image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in grayscale image
	faces = faceCascade.detectMultiScale(gray, 1.15)

	# convert cv2 imageto PIL image
	background = Image.fromarray(image)

	for (x,y,w,h) in faces:
		# resize mask
		resized_mask = mask.resize((w,h), Image.ANTIALIAS)
		# define offset for mask
		offset = (x,y)
		# pask mask on background
		background.paste(resized_mask, offset, mask=resized_mask)

	# return background as cv2 image
	return np.asarray(background)

# VideoCapture object
cap = cv2.VideoCapture(cv2.CAP_ANY)

while True:
	# read return value and frame
	ret, frame = cap.read()

	if ret == True:
		# show frame with thug life mask
		cv2.imshow('Live', thug_mask(frame))

		# chck if esc key is pressed
		if cv2.waitKey(1) == 27:
			break

# release cam
cap.release()
# destroy all open opencv windows
cv2.destroyAllWindows()