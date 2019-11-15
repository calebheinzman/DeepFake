import os
import cv2
from PIL import Image

dataPath = 'Data/originaltimit/'
subjectList = os.listdir(dataPath)
videos = []
for subject in subjectList:
    videoPath = '/video'
    fullVideoPath = dataPath + subject + videoPath
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

for video in videos:

    for frame in video:
        img = cv2.imread(frame[0])
        img = Image.fromarray(img)
    print(frame)
