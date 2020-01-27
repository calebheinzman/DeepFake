import csv
import os
import cv2
from PIL import Image
from torchvision import transforms
import torch
from SupportingsFiles import InceptionModel

#Get list of all frames
dataPath = 'Data/originaltimit/'
subjectList = os.listdir(dataPath)
videos = []
for subject in subjectList:
    videoPath = '/video'
    fullVideoPath = dataPath + subject + videoPath
    if '.DS_Store' not in subject:
        videoList = os.listdir(fullVideoPath)
        for video in videoList:
            framePath = fullVideoPath + '/' + video
            if '.DS_Store' not in video:
                frameList = os.listdir(framePath)
                if "head" not in video:
                    frames = []
                    for frame in frameList:
                        fullFramePath = framePath+'/'+frame
                        frames.append([fullFramePath,subject,video,frame,'0'])
                    videos.append(frames)

#Load the MobileNet Model
model = InceptionModel.inception_v3(pretrained=True)
model.eval()

count = 0
videoCount = 0;
for videoCount in range (0,159):
    frameFeatures = []
    for frame in videos[videoCount]:

        img = cv2.imread(frame[0])
        img = Image.fromarray(img)

        #Preprocessing to format and normalized the image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        #Gets Feature Vector
        with torch.no_grad():
            output = model(input_batch)

        #Append Frame feature to list for entire video
        frameFeatures.append(output[0].tolist())

    #Write Frame Features to Csv File
    csvPath = 'InceptionFeatureVectors/'+frame[4] + '/'
    with open(csvPath+frame[1] + '-' + frame[2]+'.csv', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(frameFeatures)
    print(frame)
    count = count + 1
    print(count)
