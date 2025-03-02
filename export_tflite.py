#!python

"""Export the model to tflite format."""
import tensorflow as tf
import time

model = tf.keras.models.load_model('saved')

x = tf.keras.Input(shape=[160,240,3], batch_size=1)
y = model(x)
model = tf.keras.Model(inputs=x, outputs=y)

print(model.summary())

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save the model.
with open('model_fp16.tflite', 'wb') as f:
  f.write(tflite_model)
