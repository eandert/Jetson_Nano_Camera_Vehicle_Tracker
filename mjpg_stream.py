import cv2
import frame_by_frame_tracker
import time
import exifread
from multiprocessing import Process, Queue
import os

def readImagesFromStream(q):
    # Fill in your details here to be posted to the login form.
    # cap = cv2.VideoCapture(0)                                      # Uncomment to use Webcam
    os.chdir("A:\\Users\edwar\Downloads")
    cap = cv2.VideoCapture("2018_04_05_200511.MOV")  # Local Stored video detection - Set input video
    frame_width = int(cap.get(3))  # Returns the width and height of capture video
    frame_height = int(cap.get(4))
    # Set out for video writer

    currentTime = 0.0
    fps = 15.0

    while True:  # Load the input frame and write output frame.
        ret, frame_read = cap.read()  # Capture frame and return true if frame present
        # For Assertion Failed Error in OpenCV
        if not ret:  # Check if frame present otherwise he break the while loop
            break
        else:
            q.put([frame_read, currentTime])
            currentTime += 1.0/fps
            time.sleep(1.0/fps)
            print("Producer: Sent an image at ", currentTime)

if __name__ == '__main__':
    q = Queue()

    producer = Process(target=readImagesFromStream, args=(q, ))
    producer.start()

    consumer = Process(target=frame_by_frame_tracker.processImages, args=(q, ))
    consumer.start()