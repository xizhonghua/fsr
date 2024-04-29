#!python

import glob
import tensorflow as tf
import random
import sys

scale_factor = 3

def build_model(scale_factor=3):
  inputs = tf.keras.Input(shape=[None, None, 3])
  x0 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
  layers = 3
  x = x0
  for _ in range(layers):
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
  
  x1 = tf.keras.layers.Conv2D(3 * scale_factor **2, 3, padding='same')(x + x0)
  outputs = tf.nn.depth_to_space(x1, scale_factor)  
  return tf.keras.Model(inputs, outputs)

def scheduler(epoch, lr):
  if epoch < 5:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def get_model():
    model = build_model()
    model.summary()
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=loss_fn)
    return model

def get_paths(dataset_name, id):
    hr = glob.glob(f'{dataset_name}/{id}/*_hr_*.png')
    lr = glob.glob(f'{dataset_name}/{id}/*_lr_*.png')
    hr.sort()
    lr.sort()
    
    if not hr or not lr: 
        return None, None
    if len(hr) != len(lr): 
        return None, None

    return lr, hr

def load_image(path):
  raw = tf.io.read_file(path)
  tensor = tf.io.decode_image(raw)
  tensor = tf.cast(tensor, tf.float32) / 255.0
  return tensor

def load_image_pair(x_path, y_path):
    return load_image(x_path), load_image(y_path)

# @tf.function
# def preprocess(x, y):
#   height = 360 
#   width = 640
#   x_sr = tf.image.resize(
#     x,
#     [height * scale_factor, width * scale_factor],
#     method='nearest'
#   )

#   x_cropped_sr, y_cropped = tf.image.random_crop(value=[x_sr, y], size=(256, 256))

#   x_cropped = tf.image.resize(
#     x_cropped_sr,
#     [height, width],
#     method='nearest'
#   )

#   return x_cropped, y_cropped

def load_dataset(x_paths, y_paths):
  dataset = tf.data.Dataset.from_tensor_slices((x_paths, y_paths))
  dataset = dataset.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
  # dataset = dataset.map(preprocess)
  dataset = dataset.batch(32)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset

def main():
  # tf.keras.backend.set_floatx('float16')

  dataset_name = 'dataset_360_v1'
  folders = glob.glob(f'{dataset_name}/*')
  x_paths = []
  y_paths = []
  for folder in folders:
    id = folder.split('/')[-1]
    lr, hr = get_paths(dataset_name, id)
    if lr == None or hr == None: continue
    x_paths += lr
    y_paths += hr
  print(f'total images = {len(x_paths)}')
  dataset = load_dataset(x_paths, y_paths)

  r = random.Random(42)
  model = get_model()
  max_iter = 100
  rpt = 1
  callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
  for i in range(max_iter):
    model.fit(dataset, epochs=i*rpt+rpt, initial_epoch=i*rpt, callbacks=[callback])
    model.save('saved')

if __name__ == '__main__':
    main()