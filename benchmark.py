#!python

import tensorflow as tf
import time

model = tf.keras.models.load_model('saved')
print(model.summary)

input_image = tf.ones([1, 360, 640, 3])

now = time.time()
runs = 100
for _ in range(runs):
    output_image = model(input_image)

avg = (time.time() - now) / runs

print(f"avg runtime = {avg}")