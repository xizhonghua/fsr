#!python

"""
Usage:
  ./run_video.py -i input.mp4 -o output.mp4

"""
import argparse
import tensorflow as tf
import cv2
import numpy as np
import sys
from vidgear.gears import WriteGear

parser = argparse.ArgumentParser(description="Run video.")
parser.add_argument("-i", "--input")
parser.add_argument("-o", "--output")
parser.add_argument("-v", "--visualize", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--height", default=720)
args = parser.parse_args()

print(args)
print(args.visualize)

model = tf.keras.models.load_model("saved")

def process_frame(lr):
  lr = tf.expand_dims(lr, axis=0)
  lr = tf.image.convert_image_dtype(lr, dtype=tf.float32, saturate=True)
  sr = model(lr)
  sr = tf.image.convert_image_dtype(sr, dtype=tf.uint8, saturate=True)
  sr = tf.squeeze(sr, axis=0)
  return sr.numpy()

cap = cv2.VideoCapture(args.input)

# get cap property 
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"{width}*{height}, {fps=}")

output_params = {"-vcodec": "libx264", "-crf": 15, "-preset": "fast", "-input_framerate":fps, "-r":fps, "-pix_fmt": "yuv420p"}

writer = WriteGear(output = args.output, compression_mode = True, logging = True, **output_params)
 

if args.visualize:
  cv2.namedWindow("lr | sr", cv2.WINDOW_NORMAL) 

while (cap.isOpened()):
  ret, lr_frame = cap.read()

  if not ret: break

  sr_frame = process_frame(lr_frame)

  preview_h = args.height
  preview_w = int(sr_frame.shape[1] * preview_h * 1.0 / sr_frame.shape[0])

  if args.visualize:
    lr_preview = cv2.resize(lr_frame, (preview_w, preview_h))
    sr_preview = cv2.resize(sr_frame, (preview_w, preview_h))
    frame = np.vstack((lr_preview, sr_preview))
    cv2.imshow("lr | sr", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  writer.write(sr_frame)
 
cap.release()
writer.close()

if args.visualize:
  cv2.destroyAllWindows()