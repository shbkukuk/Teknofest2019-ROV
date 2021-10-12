
#python calisankodobje.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import picamera
from picamera.array import PiRGBArray
from picamera import PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor",]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(2.0)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

        image = frame.array	# grab the frame from the threaded video stream and resize it

        (h, w) = image.shape[: 2]
        #print(h)
        resized_img = cv2.resize(image, (300,300), interpolation = cv2.INTER_AREA)
        #print("22")
        blob = cv2.dnn.blobFromImage(resized_img,0.007843,(300, 300),127.5)
        net.setInput(blob)
        detections = net.forward()


        for i in np.arange(0,detections.shape[2]):

            confidence = detections[0, 0, i, 2]
            if confidence > args["confidence"]:

                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
			confidence * 100)
                cv2.rectangle(image, (startX, startY), (endX, endY),
			COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

	# update the FPS counter
        rawCapture.truncate(0)
        #fps.update()



# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()