# Nvidia Jetson Nano Traffic Cam and Forward Collision Detection Code

This repository contains a YoloV4/Darknet based image classifier coded to run onboard the Nvidia Jetson Nano platform at approximately 10 FPS. The code includes a frame by frame tracker that keeps IDs of vehicles mostly constant. It display the output to the web browser via flask for debugging/demo.

This repository is still in the works so please bear with the bad instructions right now as I update them.

## Preparing the Jeston Nano
Rather than re-inventing the wheel, I would suggest you follow the instructions outlined here to set up your Nvidia Jetson Nano for image processing. We will not need tensorflow, however we do require OpenCV and SkiKitLearn so make sure you get through that part. I find this guide is the best - you may be able to follow another one. https://www.pyimagesearch.com/2020/03/25/how-to-configure-your-nvidia-jetson-nano-for-computer-vision-and-deep-learning/

After installing all the necessary prerequisites including OpenCV and scikitlearn, we can now follow the instructions to install darknet:
https://jkjung-avt.github.io/yolov4/

Finally clone this repository ont your machine and set the necessary variables through the command line. 

## Preparing a Windows machine
Install Darknet, Yolo, and OpanCV. I used this method with success on my machine that has a GTX660: https://youtu.be/5pYh1rFnNZs

Install the necessary python libraries.

Finally clone this repository ont your machine and set the necessary variables through the command line.

## Preparing a Linux machine
We will not go into details here as the steps are very similar to the Jetson Nano instrucitons. However the gist of the instructions is to install 

## Downloading the AVI files if you do not have a camera on the Jetson Nano or are wantign to run from pre-recorded footage

Please download the following files for use with this repo:
[Freeway Video]: https://drive.google.com/file/d/12MKBTURDOkKL8O1F8-N4G40rWjp_SmGs/view?usp=sharing
[In Car Video]: https://drive.google.com/file/d/1M1roYX4DFLg403jTQiz0ZsQ9qnwvytgo/view?usp=sharing