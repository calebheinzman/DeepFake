import os

#Get List of Vidoes
dataPath = 'Data/deepfaketimit/DeepfakeTIMIT/lower_quality'
folderList = os.listdir(dataPath)
folderList.remove('.DS_Store')
filePathList = []
fileType = '.avi'
for folder in folderList:
    videoPath =dataPath+'/'+folder
    fileList = os.listdir(videoPath)
    for file in fileList:
        if fileType in file:
            file = file.replace(fileType,'')
            filePathList.append([folder,file,fileType])

for i in filePathList:
    print(i)