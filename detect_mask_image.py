# imports
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Set if memory growth should be enabled for a PhysicalDevice.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
ap.add_argument('-f', '--face', type=str,
                default='face_detector',
                help='path to face detector model directory')
ap.add_argument('-m', '--model', type=str,
                default='mask_detector.model',
                help='path to trained face mask detector model')
ap.add_argument('-c', '--confidence', type=float, default=0.5,
                help='minimum probability to filter weak detections')
ap.add_argument('--cfg', type=str,
                default='config/imdim.cfg',
                help='path to image size file')
args = vars(ap.parse_args())

IMG_RESIZE = tuple(np.loadtxt(args['cfg']).astype('int'))

# load our saved model
print('[INFO] loading face detector model...')
prototxtPath = os.path.sep.join([args['face'], 'deploy.prototxt'])
weightsPath = os.path.sep.join([args['face'],
                                'res10_300x300_ssd_iter_140000.caffemodel'])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model
print('[INFO] loading face mask detector model...')
model = load_model(args['model'])

# load the input imagem from disk, clone it, and grab the image spatial
# dimensions
image = cv2.imread(args['image'])
orig = image.copy()
(h, w) = image.shape[:2]

# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1., (300, 300),
                             (104., 177., 123.))

# pass the blob through the network and obtain the face detections
print('[INFO] computing face detections...')
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(detections.shape[2]):
    # extract the confidence (i.e., probability) associated with
    # the detection
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the confidence is
    # greater than the minimun confidence
    if confidence > args['confidence']:
        # compute the (x, y)-coordinates of the bounding box for
        # the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')

        # ensure the bounding boxes fall within the dimensions of
        # the frame
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # extract the face ROI, convert it from BGR to RGB channel
        # ordering, resize it to IMG_RESIZE, adn preprocess it
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, IMG_RESIZE)
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # pass the face through the model to determine if the face
        # has a mask or not
        (mask, withoutMask) = model.predict(face)[0]

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = 'mask' if mask > withoutMask else 'no mask'
        color = (0, 255, 0) if label =='mask' else (0, 0, 255)

        # include the probability in the label
        label = f'{label}: {max(mask, withoutMask) * 100: .2f}'

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# show the output image
cv2.imshow('Output', image)
cv2.waitKey(0)
