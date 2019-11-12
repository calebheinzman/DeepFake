import csv
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn

#Load the MobileNet Model
model = torch.hub.load('pytorch/vision:v0.4.2', 'mobilenet_v2', pretrained=True)
model.eval()

# File Name of VIdeo
filePath = 'Data/'
file = 'sa1-video-fram1'
fileType = '.avi'
fileName = filePath+file+fileType

#Read Video
cap = cv2.VideoCapture(fileName)

#Get each individual frame from video
success = 1
images = []
while success:
    # vidObj object calls read
    # function extract frames
    success, image = cap.read()
    images.append(image)

frameFeatures = []

#Get Feature Vector for each frame
for i in range(images.__len__()-1):
    #Get the frame from list
    image = np.array(images[i])
    #Convert array to image type to work with preprocessing
    image = Image.fromarray(image)

    #Preprocessing to format and normalized the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    #Removes Output Layer
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier
    #Gets Feature Vector
    with torch.no_grad():
        output = model(input_batch)

    #Append Frame feature to list for entire video
    frameFeatures.append(output[0].tolist())

#Write Frame Features to Csv File
with open(file+'.csv', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(frameFeatures)
