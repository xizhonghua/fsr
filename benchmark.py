#!python

import argparse
import tensorflow as tf
import time

parser = argparse.ArgumentParser(description="Benchmark Keras models.")
parser.add_argument("-m", "--model", default="saved_model")
parser.add_argument("-b", "--batch_size", type=int, default=1)
parser.add_argument("--width", type=int, default=320)
parser.add_argument("--height", type=int, default=240)
args = parser.parse_args()

model = tf.saved_model.load(args.model)
input_image = tf.ones([args.batch_size, args.height, args.width, 3])

now = time.time()
runs = 100
print(f"Benchmark {input_image.shape} for {runs=}")
for _ in range(runs):
    _ = model.serve(input_image)

avg = (time.time() - now) / runs

print(f"avg runtime = {avg*1000} ms")