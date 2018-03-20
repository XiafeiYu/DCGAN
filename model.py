# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:48:39 2018

@author: Sofie
"""
import tensorflow as tf
import numpy as np
import scipy.misc
import os
#class model():
#    def __init__(self, learning_rate = 0.0002, beat1 = 0.5, input_height, input_width, 
#                 output_height = 64, output_width = 64, d_first_dim = 64,g_last_dim = 128, 
#                 batch_size = 128, pic_dim = 3):
#        self.learning_rate = learning_rate
#        self.beat1 = beat1
#        self.input_height = input_height
#        self.input_width = input_width
#        self.output_height = output_height
#        self.output_width = output_width
#        self.d_first_dim = d_first_dim
#        self.g_last_dim = g_last_dim
#        self.batch_size = batch_size
#        self.pic_dim = pic_dim

d_first_dim = 64
g_last_dim = 64
        
    #convolution
def conv2d(x, sp, name):
    with tf.variable_scope(name):
    #shape = [height，width，input_channel，output_channel]
        w = tf.get_variable('w', shape = sp, initializer = tf.truncated_normal_initializer(stddev = 0.02))
        conv = tf.nn.conv2d(x, w, strides = [1,2,2,1], padding = 'SAME')
        b = tf.get_variable('b', shape = sp[-1], initializer = tf.zeros_initializer())
#    return tf.add(conv, b)
        return   tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
    #deconvolution
def deconv2d(x, output_shape, shape, name):
    with tf.variable_scope(name):
    # shape = [height，width，output_channel，input_channel]
        w = tf.get_variable('w', shape = shape, 
                            initializer = tf.truncated_normal_initializer(stddev = 0.02))
#                                     initializer = tf.truncated_normal_initializer(stddev = 0.02))
        deconv = tf.nn.conv2d_transpose(x, w, output_shape = output_shape, strides = [1,2,2,1])
        b = tf.get_variable('b', shape = output_shape[-1], initializer = tf.zeros_initializer())
        return tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
    #batch normalization
def batch_norm(x, training, name):
    with tf.variable_scope(name):
#        if re_use:
#            scope.reuse_variables()
        norm = tf.contrib.layers.batch_norm(x, decay = 0.9, updates_collections=None, reuse=tf.AUTO_REUSE,
                                        scope = name, scale = True, epsilon = 1e-5, is_training = training)
    return norm
#    linear      
def linear(x, y_dim, name):
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape = [int(x.shape[1]), y_dim], 
                            initializer = tf.random_normal_initializer(stddev=0.02))
#                              tf.random_normal_initializer(stddev=0.02))
#    b = tf.Variable('b',tf.zeros(shape = [y_dim]))
        b = tf.get_variable('b', shape = [y_dim], initializer = tf.zeros_initializer())
        return tf.add(tf.matmul(x, w),b)

#    D
def discriminator(x, batch_size, re_use):
    with tf.variable_scope("discriminator") as scope:
        if re_use:
            scope.reuse_variables()

    #    norm1 = batch_norm(conv2d(x, [5, 5, int(x.shape[-1]), d_first_dim], 'conv_1'), True)
        norm1 = conv2d(x, [5, 5, int(x.shape[-1]), d_first_dim], 'd_conv_1')
        d_conv1 = tf.nn.leaky_relu(norm1, alpha = 0.2)

        
        norm2 = batch_norm(conv2d(d_conv1, [5, 5, d_first_dim, d_first_dim*2], 'd_conv_2'), True, 'd_n1')
        d_conv2 = tf.nn.leaky_relu(norm2, alpha = 0.2)

    
        norm3 = batch_norm(conv2d(d_conv2, [5, 5, d_first_dim*2, d_first_dim*4], 'd_conv_3'), True, 'd_n2')
        d_conv3 = tf.nn.leaky_relu(norm3, alpha = 0.2)

        
        norm4 = batch_norm(conv2d(d_conv3, [5, 5, d_first_dim*4, d_first_dim*8], 'd_conv_4'), True, 'd_n3')
        d_conv4 = tf.nn.leaky_relu(norm4, alpha = 0.2)

    
        d_conv5 = linear(tf.reshape(d_conv4,[batch_size,-1]), 1, 'd_liner')
        return tf.nn.sigmoid(d_conv5), d_conv5
#    G    
def generator(x, output_height, output_width, batch_size, training, re_use):
    with tf.variable_scope("generator") as scope:
        if re_use:
            scope.reuse_variables()
        projected = linear(x, (output_height//16)*(output_width//16)*(8*g_last_dim), 'g_liner')
        norm1 = tf.reshape(projected, [-1, output_height//16, output_width//16, 8*g_last_dim])
        norm1 = batch_norm(norm1, training, 'g_n1')
        g_decov1 = tf.nn.relu(norm1)
        
        norm2 = batch_norm(deconv2d(g_decov1, [batch_size, output_height//8, output_width//8, 4*g_last_dim],
                                    [5, 5, 4*g_last_dim, 8*g_last_dim], 'g_deconv_1'), training, 'g_n2')
        g_decov2 = tf.nn.relu(norm2)
        
        norm3 = batch_norm(deconv2d(g_decov2, [batch_size, output_height//4, output_width//4, 2*g_last_dim],
                                    [5, 5, 2*g_last_dim, 4*g_last_dim], 'g_deconv_2'), training, 'g_n3')
        g_decov3 = tf.nn.relu(norm3)
    
        norm4 = batch_norm(deconv2d(g_decov3, [batch_size, output_height//2, output_width//2, g_last_dim],
                                    [5, 5, g_last_dim, 2*g_last_dim], 'g_deconv_3'), training, 'g_n4')
        g_decov4 = tf.nn.relu(norm4)        
        
        norm5 = deconv2d(g_decov4, [batch_size, output_height, output_width, 3],
                                    [5, 5, 3, g_last_dim], 'g_deconv_4')
        g_decov5 = tf.nn.tanh(norm5) 
    return g_decov5

def loss_function(x,y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = x, labels = y))

def optimal(loss,learn_rate, beta1, variable):
    return tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.5).minimize(loss, var_list = variable)

    
#def sampler(x, output_height, output_width, batch_size):
#    with tf.variable_scope("generator") as scope:
#        scope.reuse_variables()
#        projected = linear(x, output_height//16*output_width//16*8*g_last_dim, 'g_liner')
#        norm1 = tf.reshape(projected, [-1, output_height//16, output_width//16, 8*g_last_dim])
#        norm1 = batch_norm(norm1, False, name = 'g_n1')
#        g_decov1 = tf.nn.relu(norm1)
#        
#        norm2 = batch_norm(deconv2d(g_decov1, [batch_size, output_height//8, output_width//8, 4*g_last_dim],
#                                    [5, 5, 4*g_last_dim, 8*g_last_dim], 'deconv_1'), False, name = 'g_n2')
#        g_decov2 = tf.nn.relu(norm2)
#        
#        norm3 = batch_norm(deconv2d(g_decov2, [batch_size, output_height//4, output_width//4, 2*g_last_dim],
#                                    [5, 5, 2*g_last_dim, 4*g_last_dim], 'deconv_2'), False, name = 'g_n3')
#        g_decov3 = tf.nn.relu(norm3)
#    
#        norm4 = batch_norm(deconv2d(g_decov3, [batch_size, output_height//2, output_width//2, g_last_dim],
#                                    [5, 5, g_last_dim, 2*g_last_dim], 'deconv_3'), False, name = 'g_n4')
#        g_decov4 = tf.nn.relu(norm4)        
#        
#        norm5 = deconv2d(g_decov4, [batch_size, output_height, output_width, 3],
#                                    [5, 5, 3, g_last_dim], 'deconv_4')
#        g_decov5 = tf.nn.tanh(norm5) 
#    return g_decov5