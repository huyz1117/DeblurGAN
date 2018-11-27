# Copyright (c) 2018 by huyz. All Rights Reserved.

import tensorflow as tf
import numpy as np

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, x*leak)

def concat(tensor, axis):
    return tf.concat(tensor, axis)

def conv2d(inputs, output_dim, filter_size=3, strides=1,
            padding="SAME", stddev=0.02, reuse=False, name="conv2d"):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if reuse==True: scope.reuse_variables()
        W = tf.get_variable(name="W", shape=[filter_size, filter_size, inputs.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable(name="b", shape=[output_dim], initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(inputs, W, strides=[1, strides, strides, 1], padding=padding)
        conv = tf.nn.bias_add(conv, b)

        return conv

def deconv2d(inputs, output_dim, filter_size, strides,
            stddev=0.02, padding="SAME", use_bias=True, initializer=None, name="deconv2d"):
    if initializer == None:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        deconv = tf.layers.conv2d_transpose(inputs, output_dim, [filter_size, filter_size],
                        strides=[strides, strides], padding=padding, use_bias=use_bias)
        return deconv

def fc_layer(inputs, output_dim, activation="linear", stddev=0.02, name=None):
    shape = inputs.get_shape().as_list()

    with tf.variable_scope(name or "Linear", reuse=tf.AUTO_REUSE):
        W = tf.get_variable(name="W", shape=[shape[1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable(name="b", shape=[output_dim], initializer=tf.zeros_initializer())
        result = tf.matmul(inputs, W) + b

        if activation == "tanh":
            result = tf.nn.relu(result)
        elif activation == "linear":
            result = result
        elif activation == "sigmoid":
            result = tf.nn.sigmoid(result)

        return result

def res_block(inputs, output_dim, filter_size, padding="SAME", name="res_block"):
    short_cut = inputs

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv = conv2d(inputs, output_dim, name=name+"/conv1")
        conv = tf.contrib.layers.instance_norm(conv)
        conv = tf.nn.relu(conv)

        conv = conv2d(conv, output_dim, name=name+"/conv2")
        conv = tf.contrib.layers.instance_norm(conv)

        conv = tf.identity(short_cut+conv, name="residual_block_output")
        return conv

def generator(inputs, ngf=64, num_block=9):
    x = inputs
    count = 1

    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        """
        downsampling: conv layer
        """
        with tf.variable_scope("head"+str(count), reuse=tf.AUTO_REUSE):
            x = conv2d(x, ngf, filter_size=7, strides=1, padding="SAME", name="g_conv1")
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.relu(x)
            count = count + 1

        num_down = 2
        for i in range(num_down):
            mult = 2 ** (i + 1)
            with tf.variable_scope("head"+str(count), reuse=tf.AUTO_REUSE):
                x = conv2d(x, ngf*mult, filter_size=3, strides=2, padding="SAME")
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.relu(x)
            count = count + 1

        for i in range(num_block):
            x = res_block(x, ngf*mult, filter_size=3, name="res_block"+str(count))
            count = count + 1

        """
        upsampling: deconv layer
        """
        num_up = 2
        for i in range(num_up):
            mult = 2 ** (num_up - i)
            with tf.variable_scope("head"+str(count), reuse=tf.AUTO_REUSE):
                x = deconv2d(x, int(ngf*mult/2), filter_size=3, strides=2, padding="SAME")
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.relu(x)
            count = count + 1

        """
        output layer
        """
        with tf.variable_scope("out", reuse=tf.AUTO_REUSE):
            x =conv2d(x, 3, filter_size=7, strides=1, padding="SAME")
            x = tf.nn.tanh(x)

        """
        skip connection
        """
        out = tf.add(x, inputs)
        return out

def discriminator(inputs, ndf=64, num_layers=3):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("h0", reuse=tf.AUTO_REUSE):
            x = conv2d(inputs, ndf, filter_size=4, strides=2, padding="SAME")
            x = lrelu(x)

        nf_mult, nf_mult_prev = 1, 1
        for n in range(1, num_layers+1):
            nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
            with tf.variable_scope("h"+str(n), reuse=tf.AUTO_REUSE):
                x = conv2d(x, ndf*nf_mult, filter_size=4, strides=2, padding="SAME")
                x = tf.contrib.layers.batch_norm(x)
                x = lrelu(x)

        nf_mult_prev, nf_mult = nf_mult, min(2**num_layers, 8)
        with tf.variable_scope("h"+str(num_layers+1), reuse=tf.AUTO_REUSE):
            x = conv2d(x, ndf*nf_mult, filter_size=4, strides=1, padding="SAME")
            x = tf.contrib.layers.batch_norm(x)
            x = lrelu(x)

        """
        build output layer
        """
        with tf.variable_scope("h_out", reuse=tf.AUTO_REUSE):
            x = conv2d(x, ndf*nf_mult, filter_size=4, strides=1, name="conv")
            x = tf.contrib.layers.flatten(x)
            x = fc_layer(x, 1024, activation="tanh", name="fc1")
            out = fc_layer(x, 1, activation="linear", name="output")
            return out


if __name__ == '__main__':
    test_input = np.ones([1, 256, 256, 3], dtype=np.float32)
    test_input = tf.constant(test_input, dtype=tf.float32)

    disc = discriminator(test_input)
    tf_vars = [var for var in tf.trainable_variables() if "disc" in var.name]
    for idx, var in enumerate(tf_vars):
        print(var)
    print(disc)

    gen = generator(test_input)
    tf_vars = [var for var in tf.trainable_variables() if "gen" in var.name]
    for idx, var in enumerate(tf_vars):
        print(var)
    print(gen)
