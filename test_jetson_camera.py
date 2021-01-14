from flask import Response
from flask import Flask
from flask import render_template
import threading
import nanocamera as nano
import cv2
import socket

app = Flask(__name__)

outputFrame = None
lock = threading.Lock()

def capture_image():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # Create the Camera instance
    camera = nano.Camera(flip=2, width=1280, height=720, fps=30)
    print('CSI Camera ready? - ', camera.isReady())
    # Now that the init is done, lets get our IP and print to terminal so we know where to connect
    myip = [l for l in (
        [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [
            [(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in
             [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]
    print("Please connect to http://" + str(myip) + ":5000 to see the video feed.")
    while camera.isReady():
        # read the camera image
        frame_read = camera.read()
        # Setup the variables we need, hopefully this function stays active
        with lock:
            # get the lock and set the image to the current one
            outputFrame = frame_read

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    # start a thread that will perform motion detection
    t = threading.Thread(target=capture_image, args=())
    t.daemon = True
    t.start()

    # Startup the web service
    app.run(host='0.0.0.0')