#!python

import tensorflow as tf
import sys

args = sys.argv[1:]
input_image = tf.keras.utils.img_to_array(tf.keras.utils.load_img(args[0]))
input_image = tf.expand_dims(input_image, axis=0)
input_image = input_image / 255.0

model = tf.keras.models.load_model('saved')
print(model.summary)

output_image = model(input_image)
output_image = tf.image.convert_image_dtype(output_image, dtype=tf.uint8, saturate=True)
output_image = tf.squeeze(output_image, axis=0)
print(output_image.shape)

tf.keras.utils.save_img(args[1], output_image, scale=False)
