import tensorflow as tf
import numpy as np
from tf.keras.applications.vgg16 import VGG16
from tf.keras.models import Model

image_shape = [256, 256, 3]

def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true, y_pred))

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include=False, weights="imagenet", input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output)
    loss_model.trainable = False
    return tf.reduce_mean(tf.square(loss_model(y_true), loss_model(y_pred)))

def wassertein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true, y_pred)

def gradient_penalty_loss(y_true, y_pred, average_samples):
    gradients = tf.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
    gradient_penalty = tf.square(1 - gradient_l2_norm)
    return tf.reduce_mean(gradient_penalty)

    
