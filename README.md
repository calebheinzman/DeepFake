# Setup:
1. Run `pip install --requirement requirements.txt`

# Purpose
The purpose of this project is to input videos and get corresponding feature vectors. We are using pytorch's pretrained models 
on the ImageNet dataset. This dataset is located at <http://www.image-net.org>. This dataset conists of 14 million images and labels. 
It is used to train these models for classification. We are taking the layer, that is 1280 features in size, just before the output layer and using it as features. 
We are then storing these features in a csv file corresponding to each video.

# The Data

All data is in the Data directory. 

## Original Dataset
The originaltimit directory is where the original data set is located. The orignal dataset can be found here: <http://conradsanderson.id.au/vidtimit/>. 
Each directory withing the orginaltimit directory corresponds to the videos from one subject. 
For example: 'fadg0' is one human subject. Within the subject folder are an audio and video folder. For this project we 
are only interested in the video folder. Within the video folder are more folders. This time the folders corresponds to 
the given phrase the subject says. For example: In 'sa1' the subject says "She had your dark suit in greasy wash water all year".
Within this phrase folder are the individual frames for each video.

## DeepFake Dataset
The deepfaketimit directory is where the deep fake videos are stored. These videos were taken from <https://www.idiap.ch/dataset/deepfaketimit>. 
This dataset is divided into both lower and higher qualiy videos. For the sake of this project, we use the lower quality videos. 
Within the lower_quality folder are more folders dividing the videos by the subject used from the original video. For example: 'fadg0' is one human subject which the fake is based on. 
Within the subject folder are all the videos and audio files corresponding to that subject. The videos are titled with the 
phrase that is used, then a "-video-", then the subject who's face is used to create the deepfake. For example: In folder 
"fadg0" the first video file is "sa1-video-fram1.avi". The original video used was "faqg0". The phrase that the subject is saying 
is "sa1". The face the is trying to be imposed on fadg0 is "fram1".

# Running

## MobileNet
- Running MobileNet.py will apply the mobilenet model to the fake videos.
### Output
- The output will be a csv file containing the feature vectors corresponding to each video. The output for the mobilenet applied 
on fake videos is located in "FeatureVectors/1". Each csv file is named as follows "originalvideo-phrase-video-imposedvideo". 
For example: in the first file "fadg0-sa1-video-fram1.csv". "Fadg0" is the name of the orginal video used. "sa2" is the name of the 
phrase that is spoken. And "fram1" is the video with the subjects face that will be imposed over "fadg0". Each line in the csv corresponds 
to the feature vectors for one frame of the video.

- Running MobileNetOriginal will apply the mobilenet model to the original videos.
### Output
- The output will be a csv file containing the feature vectors corresponding to each video. The output for the mobilenet applied 
on original videos is located in "FeatureVectors/0". Each csv file is named as follows "subject-phrase". 
For example: in the first file "fadg0-sa1.csv". "Fadg0" is the name of the subject video used. "sa2" is the name of the 
phrase that is spoken. Each line in the csv corresponds to the feature vectors for one frame of the video.

## Inception
- Running Inception.py will apply the inception model to the fake videos.
### Output
- The output will be a csv file containing the feature vectors corresponding to each video. The output for the Inception applied 
on fake videos is located in "InceptionFeatureVectors/1". Each csv file is named as follows "originalvideo-phrase-video-imposedvideo". 
For example: in the first file "fadg0-sa1-video-fram1.csv". "Fadg0" is the name of the orginal video used. "sa2" is the name of the 
phrase that is spoken. And "fram1" is the video with the subjects face that will be imposed over "fadg0". Each line in the csv corresponds 
to the feature vectors for one frame of the video.

- Running InceptionOriginal will apply the inception model to the original videos.
### Output
- The output will be a csv file containing the feature vectors corresponding to each video. The output for the Inception applied 
on original videos is located in "InceptionFeatureVectors/0". Each csv file is named as follows "subject-phrase". 
For example: in the first file "fadg0-sa1.csv". "Fadg0" is the name of the subject video used. "sa2" is the name of the 
phrase that is spoken. Each line in the csv corresponds to the feature vectors for one frame of the video.




