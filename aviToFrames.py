import numpy as np
import cv2
import matplotlib.pyplot as plt

fileName = 'Data/sa1-video-fram1.avi'

# read video
cap = cv2.VideoCapture(fileName)

success = 1
images = []

while success:
    # vidObj object calls read
    # function extract frames
    success, image = cap.read()
    images.append(image)

#Shows the first Image
plt.figure()
plt.imshow(images[1])
plt.show()




