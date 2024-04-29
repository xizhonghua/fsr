#!python

import tensorflow as tf
import cv2
import numpy as np
import sys
from vidgear.gears import WriteGear

args = sys.argv[1:]

input_video = args[0]
output_video = args[1]
SCALE=args[2] if len(args) > 2 else 2

model = tf.keras.models.load_model('saved')
 
def process_frame(lr):
    lr = tf.expand_dims(lr, axis=0)
    lr = tf.image.convert_image_dtype(lr, dtype=tf.float32, saturate=True)
    sr = model(lr)
    sr = tf.image.convert_image_dtype(sr, dtype=tf.uint8, saturate=True)
    sr = tf.squeeze(sr, axis=0)
    return sr.numpy()

# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture(input_video)

# get cap property 
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
fps = cap.get(cv2.CAP_PROP_FPS)

print(f'{width}*{height}, {fps=}')

output_params = {"-vcodec":"libx264", "-crf": 15, "-preset": "fast", "-input_framerate":fps, "-r":fps, "-pix_fmt": "yuv420p"}

writer = WriteGear(output = output_video, compression_mode = True, logging = True, **output_params)
 
# Loop until the end of the video

cv2.namedWindow("lr | sr", cv2.WINDOW_NORMAL) 

while (cap.isOpened()):
 
    # Capture frame-by-frame
    ret, lr_frame = cap.read()

    if not ret: break
 
    sr_frame = process_frame(lr_frame)

    lr_frame_scaled = cv2.resize(lr_frame, (sr_frame.shape[1], sr_frame.shape[0]))
    
    frame = np.vstack((lr_frame_scaled, sr_frame))

    # Display the resulting frame
    cv2.imshow('lr | sr', frame)

    # output.write(sr_frame)
    writer.write(sr_frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# release the video capture object
cap.release()
writer.close()

# Closes all the windows currently opened.
cv2.destroyAllWindows()