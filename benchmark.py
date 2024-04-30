#!python

import argparse
import tensorflow as tf
import time

parser = argparse.ArgumentParser(description="Benchmark Keras models.")
parser.add_argument("-m", "--model", default="models/anime_v1_3x")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=360)
args = parser.parse_args()

model = tf.keras.models.load_model(args.model)
print(model.summary())

input_image = tf.ones([1, args.height, args.width, 3])

now = time.time()
runs = 100
print(f"Benchmark {input_image.shape} for {runs=}")
for _ in range(runs):
    output_image = model(input_image)

avg = (time.time() - now) / runs

print(f"avg runtime = {avg}")