import cv2
from PIL import Image
import sys

# input image path
imagePath = sys.argv[1]
# thug life meme mask image path
maskPath = "mask.png"
# haarcascade path
cascPath = "haarcascade_frontalface_default.xml"

# cascade classifier object 
faceCascade = cv2.CascadeClassifier(cascPath)

# read input image
image = cv2.imread(imagePath)
# convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in grayscale image
faces = faceCascade.detectMultiScale(gray, 1.15)

# open input image as PIL image
background = Image.open(imagePath)

# paste mask on each detected face in input image
for (x,y,w,h) in faces:

	# just to show detected faces
	cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2)
	cv2.imshow('face detected', image)
	cv2.waitKey(0)

	# open mask as PIL image
	mask = Image.open(maskPath)
	# resize mask according to detected face
	mask = mask.resize((w,h), Image.ANTIALIAS)

	# define offset for mask
	offset = (x,y)
	# paste mask on background
	background.paste(mask, offset, mask=mask)

# paste final thug life meme
background.save('out.png')