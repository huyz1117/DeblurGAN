# Copyright (c) 2018 by huyz. All Rights Reserved.

import tensorflow as tf
from vgg19 import Vgg19
from ops import *
import os
from utils import load_images

image_shape = [1, 256, 256, 3]
lr = 0.0001

class DeblurGAN():

    def name(self):
        return "DeblurGAN"

    def __init__(self, sess):
        self.sess = sess
        self.args = args
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.vgg_path = args.vgg_path
        self.decay_step = args.decay_step
        self.mode = args.mode
        self.epoch = args.epoch

    def build_model(self):
        data = load_images("./images/train", n_images)
        y_train, x_train = data["B"], data["A"]

        for index in range(int(x_train.shape[0] / self.batch_size)):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]

        x = image_blur_batch
        label = image_full_batch

        self.gen_img = generator(x)
        self.real_prob = discriminator(label)
        self.fake_prob = discriminator(self.gen_img)

        epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
        interpolated_input = epsilon * label + (1 - epsilon) * self.gen_img
        gradient = tf.gradients(discriminator(interpolated_input), [interpolated_input])[0]
        GP_loss = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_mean(tf.square(gradient), axis=[1, 2, 3])) - 1))

        d_loss_real = -tf.reduce_mean(self.real_prob)
        d_loss_fake = tf.reduce_mean(self.fake_prob)

        self.saver = tf.train.Saver()

        if self.mode == "train":
            self.vgg_net = Vgg19(self.vgg_path)
            self.vgg_net.build(tf.concat([label, self.gen_img], axis=0))
            self.content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.vgg_net.relu3_3[self.batch_size:] - self.vgg_net.relu3_3[:self.batch_size]), axis = 3))

            self.D_loss = d_loss_real + d_loss_fake + 10.0 * GP_loss
            self.G_loss = d_loss_fake + 100.0 * self.content_loss

            t_vars = tf.trainable_variables()
            G_vars = [var for var in t_vars if "generator" in var.name]
            D_vars = [var for var in t_vars if "discriminator" in var.name]

            lr = tf.minimum(self.learning_rate, tf.abs(2 * self.learning_rate - (self.learning_rate * tf.cast(self.epoch, tf.float32) / self.decay_step)))
            self.D_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.D_loss, var_list=D_vars)
            self.G_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.G_loss, var_list=G_vars)

            logging_D_loss = tf.summary.scalar("D_loss", self.D_loss)
            logging_G_loss = tf.summary.scalar("G_loss", self.G_loss)

        self.PSNR = tf.reduce_mean(tf.image.psnr(((self.gen_img + 1.0) / 2.0), ((label + 1.0) / 2.0), max_val = 1.0))
        self.SSIM = tf.reduce_mean(tf.image.ssim(((self.gen_img + 1.0) / 2.0), ((label + 1.0) / 2.0), max_val = 1.0))

        logging_PSNR = tf.summary.scalar("PSNR", self.PSNR)
        logging_SSIM = tf.summary.scalar("SSIM", self.SSIM)

    def save_weights(self, checkpoint_dir, step):
        model_name = self.args.model_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load_weights(self, checkpoint_dir):
        import re
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Success to read {}".format(ckpt_name))
            return False, 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("discription")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--vgg_path", type=str, default="./vgg19.npy")
    parser.add_argument("--decay_step", type=int, default=150)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--model_name", type=str, default="DeblurGAN")

    args = parser.parse_args()

    sess = tf.Session()

    test_DeblurGAN = DeblurGAN(sess)
    test_DeblurGAN.build_model()
    print(test_DeblurGAN.gen_img)
    print(test_DeblurGAN.real_prob)
