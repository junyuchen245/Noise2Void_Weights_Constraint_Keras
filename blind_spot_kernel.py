import tensorflow as tf
import numpy as np
import math, sys, os
class blind_spot(tf.keras.constraints.Constraint):
  """
  Blind Spot Kernel Constraint
    Junyu Chen
    jchen245@jhmi.edu

  Original paper:
    A. Krull, T. Buchholz and F. Jug, 
    "Noise2Void - Learning Denoising From Single Noisy Images," 
    2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 
    Long Beach, CA, USA, 2019, pp. 2124-2132, 
    doi: 10.1109/CVPR.2019.00223.
  """

  def __init__(self, filt_sz=3, ndim=2):
    self.filt_sz = filt_sz
    self.ndim = ndim

  def __call__(self, w):
    mask = np.ones(w.get_shape().as_list())
    if self.ndim == 2:
        mask[math.floor(self.filt_sz/2),math.floor(self.filt_sz/2),:,:] = 0
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    elif self.ndim == 3:
        mask[math.floor(self.filt_sz/2),math.floor(self.filt_sz/2),math.floor(self.filt_sz/2),:,:] = 0
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    else:
        mask = mask
    return w*mask

  def get_config(self):
    return {'ref_value': self.ref_value}