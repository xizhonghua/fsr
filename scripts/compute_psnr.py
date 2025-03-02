#!python
import glob
from math import log10, sqrt
import sys
import tensorflow as tf
import numpy as np
import cv2
import os

def PSNR(original, compressed): 
  mse = np.mean((original - compressed) ** 2) 
  if(mse == 0):
    return 100
  max_pixel = 255.0
  psnr = 20 * log10(max_pixel / sqrt(mse)) 
  return psnr

# args = sys.argv[1:]
# lr = cv2.imread(args[0])
# hr = cv2.imread(args[1])
# sr = cv2.resize(lr, (hr.shape[1], hr.shape[0]))
# print(PSNR(hr, sr))

def get_paths(dataset_name, id):
  hr = glob.glob(f'{dataset_name}/{id}/*_hr_*.png')
  lr = glob.glob(f'{dataset_name}/{id}/*_lr_*.png')
  hr.sort()
  lr.sort()
  
  if not hr or not lr: 
    return None, None
  if len(hr) != len(lr): 
    print(f"issue with {id}")
    return None, None

  return lr, hr

dataset_name = 'dataset_360_v1'
folders = glob.glob(f'{dataset_name}/*')
for folder in folders:
  id = folder.split('/')[-1]
  lr_paths, hr_paths = get_paths(dataset_name, id)
  if lr_paths == None or hr_paths == None: 
    print(f'Issue with folder: {id}')
    continue
  for lr_path, hr_path in zip(lr_paths, hr_paths):
    lr = cv2.imread(lr_path)
    hr = cv2.imread(hr_path)
    sr = cv2.resize(lr, (hr.shape[1], hr.shape[0]))
    psnr = PSNR(hr, sr)
    ssim = tf.image.ssim(hr, sr, 255)
    print(f"{lr_path},{hr_path},{psnr},{ssim}")
    if psnr < 35 or psnr > 50 or ssim < 0.92:
      os.rename(lr_path, lr_path + ".bak")
      os.rename(hr_path, hr_path + ".bak")
      print(f"{lr_path},{hr_path},renamed!!!")


  