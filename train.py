#!python

import argparse
import glob
import random
import sys
import tensorflow as tf

# tf.keras.config.set_dtype_policy("mixed_float16")

parser = argparse.ArgumentParser(description="Train the model.")
parser.add_argument("-s", "--scale_factor", type=int, default=3)
parser.add_argument("-f", "--filters", type=int, default=64)
parser.add_argument("--upsample_filters", type=int, default=64)
parser.add_argument("-l", "--layers", type=int, default=4)
parser.add_argument("-d", "--dataset", default="dataset_ghibli")
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("--steps_per_epoch", type=int, default=200)
parser.add_argument("-b", "--batch_size", default=16)
parser.add_argument("--lr_patch_size", type=int, default=64)
parser.add_argument("--jpeg_prob", default=0.5)
parser.add_argument("--min_jpeg_quality", default=20)
parser.add_argument("--max_jpeg_quality", default=95)
args = parser.parse_args()

class VGGFeatureMatchingLoss(tf.keras.losses.Loss):
  def __init__(self, **kwargs):
      super().__init__(**kwargs)
      self.encoder_layers = [
        "block2_conv2",
      ]
      vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
      layer_outputs = [vgg.get_layer(x).output for x in self.encoder_layers]
      self.vgg_model = tf.keras.Model(vgg.input, layer_outputs, name="VGG")
      self.vgg_model.trainable=False
      self.mae = tf.keras.losses.MeanAbsoluteError()

  def call(self, y_true, y_pred):
      y_true = tf.keras.applications.vgg19.preprocess_input(255 * y_true)
      y_pred = tf.keras.applications.vgg19.preprocess_input(255 * y_pred)
      real_feature = self.vgg_model(y_true)
      fake_feature = self.vgg_model(y_pred)
      vgg_loss = self.mae(real_feature, fake_feature)
      mae = tf.abs(y_true - y_pred)
      return tf.reduce_mean(mae) + tf.reduce_mean(vgg_loss)

class CombinedLoss(tf.keras.losses.Loss):
  def call(self, y_true, y_pred):
    mae = tf.abs(y_true - y_pred)
    # dx, dy = tf.image.image_gradients(y_true - y_pred)
    # edge_loss = tf.reduce_mean(dx ** 2 + dy ** 2)
    # return mae + 10 * edge_loss
    return mae

class ExportModelCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    self.model.export("saved_model")

class SaveModelCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    self.model.save("model.keras")

class LogImageCallback(tf.keras.callbacks.Callback):
  def __init__(self, dataset):
    super().__init__()
    self.dataset = dataset.unbatch()

  def on_epoch_begin(self, epoch, logs):
    index = 0
    for lr, hr in self.dataset.take(5):
      sr = tf.squeeze(self.model(tf.expand_dims(lr, axis=0)),axis=0)
      lr = tf.image.resize(lr, tf.shape(sr)[0:2])
      sr = tf.image.convert_image_dtype(sr, dtype=tf.uint8, saturate=True)
      lr = tf.image.convert_image_dtype(lr, dtype=tf.uint8, saturate=True)
      hr = tf.image.convert_image_dtype(hr, dtype=tf.uint8, saturate=True)
      tf.keras.utils.save_img(f'log/train_example_{epoch}_{index}_sr.png', sr)
      tf.keras.utils.save_img(f'log/train_example_{epoch}_{index}_lr.png', lr)
      tf.keras.utils.save_img(f'log/train_example_{epoch}_{index}_hr.png', hr)
      index += 1

def upsample(x, scale_factor, filters):
  if scale_factor == 2 or scale_factor == 3:
    x = tf.keras.layers.Conv2D(filters * scale_factor**2, 3, padding='same')(x)
    x = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=scale_factor))(x)
  elif scale_factor == 4:
    x = tf.keras.layers.Conv2D(filters * 2**2, 3, padding='same')(x)
    x = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=2))(x)
    x = tf.keras.layers.Conv2D(filters * 2**2, 3, padding='same')(x)
    x = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=2))(x)
  else:
    raise ValueError(f"Unsupported {scale_factor=}")
  return x

def build_model(scale_factor, filters, layers, channels=3, kernel_size=3):
  inputs = tf.keras.Input(shape=[None, None, 3])
  x0 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(inputs)
  x = prev = x0
  for _ in range(layers):
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x) + prev
    prev = x
  x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x) + x0
  x = upsample(x, args.scale_factor, args.upsample_filters)
  outputs = tf.keras.layers.Conv2D(channels, kernel_size, padding='same')(x)
  return tf.keras.Model(inputs, outputs)

def scheduler(epoch, lr):
  return lr * 0.92

def get_paths(dataset_name, id):
  hr = glob.glob(f'{dataset_name}/{id}/*.jpg')
  return hr, hr

def load_image(path):
  raw = tf.io.read_file(path)
  image = tf.io.decode_image(raw)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return image

def load_image_pair(x_path, y_path):
  return None, load_image(y_path)


def resize(image, height, width, method):
  return tf.image.resize(image, [height, width], method=method)

def random_jpeg_quality_with_probability(image, probability=0.5, min_jpeg_quality=60, max_jpeg_quality=95):
  """
  Applies tf.image.random_jpeg_quality with a given probability.

  Args:
    image: A 3-D uint8 Tensor of shape [height, width, channels] 
           with values in [0, 255] representing an image.
    probability: A float representing the probability of applying the 
                 random JPEG quality operation.
    min_jpeg_quality: An int representing the minimum JPEG quality.
    max_jpeg_quality: An int representing the maximum JPEG quality.

  Returns:
    The processed image (either with random JPEG quality or the original).
  """

  def _apply_random_jpeg_quality(img):
    return tf.image.random_jpeg_quality(img, min_jpeg_quality, max_jpeg_quality)

  # Generate a random number and check against the probability.
  apply_transform = tf.random.uniform(shape=[], minval=0, maxval=1) < probability

  # Use tf.cond to apply the transformation conditionally.
  return tf.cond(apply_transform,
                  lambda: _apply_random_jpeg_quality(image),  # True branch: Apply the transform
                  lambda: image)  # False branch: Return the original image

def preprocess(x, y):
  lr_width = lr_height = args.lr_patch_size

  crop_width = lr_width * args.scale_factor
  crop_height = lr_height * args.scale_factor
  sr_height = lr_height * args.scale_factor
  sr_width = lr_width * args.scale_factor
  
  y = tf.image.random_crop(y, (crop_height, crop_width, 3))
  x = tf.image.resize(y, size=[lr_height, lr_width], method='nearest')

  x = random_jpeg_quality_with_probability(x, probability=args.jpeg_prob)

  return x, y

def load_dataset(x_paths, y_paths):
  dataset = tf.data.Dataset.from_tensor_slices((x_paths, y_paths))
  dataset = dataset.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
  dataset = dataset.repeat()
  dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
  dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
  dataset = dataset.batch(args.batch_size)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset

def psnr(y_true, y_pred):
  return tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1.0))


def main():
  dataset_name = args.dataset
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
  model = build_model(scale_factor=args.scale_factor, 
                      filters=args.filters,
                      layers=args.layers)
  model.summary()

  # loss_fn = tf.keras.losses.MeanAbsoluteError()
  loss_fn = VGGFeatureMatchingLoss()
  opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
  model.compile(optimizer=opt, loss=loss_fn, metrics=[psnr])

  model.fit(dataset, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, callbacks=[
    tf.keras.callbacks.LearningRateScheduler(scheduler), ExportModelCallback(), LogImageCallback(dataset)])
  model.export("exported_model")

if __name__ == '__main__':
    main()