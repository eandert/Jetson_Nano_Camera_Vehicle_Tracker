# Nvidia Jetson Nano Traffic Cam and Forward Collision Detection Code

This repository contains a YoloV4/Darknet based image classifier coded to run onboard the Nvidia Jetson Nano platform at approximately 10 FPS when using Yolov4 Tiny. The code includes a frame by frame tracker that keeps IDs of vehicles mostly constant. It display the output to the web browser via flask for debugging/demo.

This repository is still in the works so please bear with the bad instructions right now as I update them.

Hardware:
 * Procesor: Nvidia Jetson Nano 4GB
 * Camera: IMX219 160

## Preparing the Jeston Nano
Rather than re-inventing the wheel, I would suggest you follow the instructions outlined here to set up your Nvidia Jetson Nano for image processing. We will not need tensorflow, however we do require OpenCV and SkiKitLearn so make sure you get through that part. I find this guide is the best - you may be able to follow another one. https://www.pyimagesearch.com/2020/03/25/how-to-configure-your-nvidia-jetson-nano-for-computer-vision-and-deep-learning/

After installing all the necessary prerequisites including OpenCV and scikitlearn, we can now follow the instructions to install darknet:
https://jkjung-avt.github.io/yolov4/

Finally clone this repository on your machine and set the necessary variables through the command line. 

## Preparing a Windows machine
Install Darknet, Yolo, and OpenCV. I used this method with success on my machine that has a GTX660: https://youtu.be/5pYh1rFnNZs

Install the necessary python libraries.

Finally clone this repository on your machine and set the necessary variables through the command line.

## Preparing a Linux machine
We will not go into details here as the steps are very similar to the Jetson Nano instrucitons. However the gist of the instructions is to install 

## Downloading the AVI files if you do not have a camera on the Jetson Nano or are wantign to run from pre-recorded footage

You can download the following pre recorded video files for use with this repo so that it can be used without a Jetson Camera:
 * [Freeway Video](https://drive.google.com/file/d/12MKBTURDOkKL8O1F8-N4G40rWjp_SmGs/view?usp=sharing)
 * [In Car Video](https://drive.google.com/file/d/1M1roYX4DFLg403jTQiz0ZsQ9qnwvytgo/view?usp=sharing)

The following are recorded outputs that you can expect from the various settings using the pre-recorded video files:
 * [Freeway Video Yolov4 Tiny](https://drive.google.com/file/d/1JwWT1EKlWOqTKZCoDnC8k81-qbyL3Y6A/view?usp=sharing)
 * [Freeway Video Yolov4](https://drive.google.com/file/d/1F_4pNioTDJ8xbgJW0E8fmnHgXNwmZDmf/view?usp=sharing)

## Running the code
There are many different command line options for running this. Those can be found by running `python jetson_camera_recognition.py --help` however below outlines some of the common ways to run this code.

To run the code on a jetson nano with darknet either in the path or in the same folder and you wish to use the camera, the following command can be used: `python jetson_camera_recognition.py`

To run the code on a jetson nano with darknet either in the path or in the same folder and you wish to use the camera and print the forward collision detection alerts, the following command can be used: `python jetson_camera_recognition.py --collisionwarning`

To indicate that darknet.py is in a different folder, you can indicate it using `python jetson_camera_recognition.py --darknetpath "~/projects/darknet/"`

If you do not have a camera on your jetson or are runnning on a different platform like windows we can replay one of the 2 included videos (see how to download above). For instance on a windows machine we can use freeway.avi by using: `python jetson_camera_recognition.py --darknetpath "C:/Yolo_v4/darknet/build/darknet/x64/" --playback "freeway.avi"`
 
If you do not have a camera on your jetson or are runnning on a different platform like windows we can replay one of the 2 included videos (see how to download above). For instance on a windows machine we can use freeway.avi by using: `python jetson_camera_recognition.py --darknetpath "C:/Yolo_v4/darknet/build/darknet/x64/" --playback "incar.avi" --collisionwarning`
 
If you are finding that tiny yolov4 (which is the default) is not accurate enough then you can use the followign command to run, this was used on my windows machine that has a larger GPU and runs at ~5 FPS however in practice running this on the Jetson nano results in ~1 FPS `python jetson_camera_recognition.py --darknetpath "C:/Yolo_v4/darknet/build/darknet/x64/" --playback "freeway.avi" --notinyyolo`

Finally if you want to runn the full yolo with the incar.avi example you can use this command (be sure to change the darknetpath to your path): `python jetson_camera_recognition.py --darknetpath "C:/Yolo_v4/darknet/build/darknet/x64/" --playback "incar.avi" --notinyyolo --collisionwarning`

The python file `test_jetson_camera.py` is included to test is the camera can be read and outputted using flask. This does not require darknet or Yolo to be installed but does require OpenCV and Flask
