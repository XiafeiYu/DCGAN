# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 10:07:25 2018

@author: Sofie
"""

import tensorflow as tf
import numpy as np
from glob import glob
import os
import numpy as np
import cv2
from PIL import Image
import math
from model import *
from preprocessing import *
#from preprocessing import *

flags = tf.app.flags
flags.DEFINE_integer('epoch', 50, 'epoch')
flags.DEFINE_float('beta1', 0.5, 'beta1')
flags.DEFINE_integer('output_height', 64, 'output height')
flags.DEFINE_integer('output_width', 64, 'output width')
flags.DEFINE_integer('batch_size', 64, 'mini batch size')
flags.DEFINE_float('learning_rate', 0.0002, 'learning rate')
flags.DEFINE_boolean('train', False, 'True for training and False for testing')
flags.DEFINE_string('raw_dataset', './celebA', 'raw dataset, we use celeA')
flags.DEFINE_string('proceed_dataset', './face', 'images after face detect and resize for training')
FLAGS = flags.FLAGS

def main(_):
#    data = []
#    data_path = face_detect('./faces')
#    file_list = glob(os.path.join('./faces', '*.jpg'))
#    for file in file_list:
#        data.append(cv2.imread(file))
    
    input_image = tf.placeholder(tf.float32, shape = [FLAGS.batch_size, 64, 64, 3])
    z = tf.placeholder(tf.float32, shape = [None, 100])

    D, D_logit = discriminator(input_image, FLAGS.batch_size, re_use = False)
    G = generator(z, FLAGS.output_height, FLAGS.output_width, FLAGS.batch_size,
                  training = True, re_use = False)
    D_, D_logit_ = discriminator(G, FLAGS.batch_size, re_use = True)
    
         
    g_loss = loss_function(D_logit_, tf.ones_like(D_))
    d_loss_real = loss_function(D_logit, tf.ones_like(D))
    d_loss_fake = loss_function(D_logit_, tf.zeros_like(D_))
    d_loss = d_loss_real + d_loss_fake
    
    sample = generator(z, FLAGS.output_height, FLAGS.output_width, FLAGS.batch_size, 
                       training = False, re_use = True)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    g_vars = [var for var in t_vars if 'generator' in var.name]
    
    opt_d = optimal(d_loss, FLAGS.learning_rate, FLAGS.beta1, d_vars)
    opt_g = optimal(g_loss, FLAGS.learning_rate, FLAGS.beta1, g_vars)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
#    sess = tf.Session()
    if FLAGS.train:
        print('you can find training generate examples in training_samples')
        if os.path.exists(FLAGS.proceed_dataset) is True:
            print('begin to load proceed images')
            data = []
            file_list = glob(os.path.join(FLAGS.proceed_dataset, '*.jpg'))
            for file in file_list:
                data.append(cv2.imread(file))
        else:
            print('the program is beginning preprocessing the raw images')
            file_path = FLAGS.raw_dataset
            data = detect_face(file_path, FLAGS.output_height, FLAGS.output_width)
        num_batch = len(data) // FLAGS.batch_size
        sample_batch = np.array(data[0: FLAGS.batch_size])
        sample_batch = np.array(sample_batch)/127.5 - 1.

#    if not os.path.exists('Model/model.ckpt.meta'):
        with tf.Session(config=run_config) as sess:
            tf.global_variables_initializer().run()
            if os.path.exists('Model/model.ckpt.meta'):
#        with tf.Session(config=run_config) as sess: 
                saver = tf.train.Saver()
                saver.restore(sess, './Model/model.ckpt')
            for epo in range(FLAGS.epoch):
                counter = 1
                for i in range(num_batch):
                    
                    input_batch = np.array(data[i*FLAGS.batch_size : (i+1)*FLAGS.batch_size])
                    input_batch = np.array(input_batch)/127.5 - 1.
                    z_batch = np.random.uniform(-1, 1, [FLAGS.batch_size, 100]).astype(np.float32)
                    
                    sess.run(opt_d, feed_dict = {input_image: input_batch, z: z_batch})                
                    sess.run(opt_g, feed_dict = {z: z_batch})               
                    sess.run(opt_g, feed_dict = {z: z_batch})
    #                print(D_logit_.eval({z:z_batch}))
    #                print(D_.eval({z:z_batch}))
    #                print(D_logit.eval({input_image:sample_batch}))
    #                print(D.eval({input_image:sample_batch}))
    #                print(d_loss_fake.eval(feed_dict = { z: z_batch }))
    #                print(d_loss_real.eval(feed_dict = { input_image: input_batch }))
    #                print(g_loss.eval(feed_dict = {z: z_batch}))
    
                    errD_fake = d_loss_fake.eval(feed_dict = { z: z_batch })
    
                    errD_real = d_loss_real.eval(feed_dict = { input_image: input_batch })
                    errG = g_loss.eval(feed_dict = {z: z_batch})
    
                    print("Epoch: [%2d] [%4d/%4d], d_loss_fake: %.8f, d_loss_real: %.8f, g_loss: %.8f" % (epo, i, num_batch, errD_fake, errD_real, errG))
                    counter += 1
                    if counter % 200 == 0:
                        samples, sample_d_loss, sample_g_loss = sess.run([sample, d_loss, g_loss], 
                                                        feed_dict = {z: z_batch, input_image: sample_batch})                        
                        
    #                    mergeImage.save('./samples/'+ str(epo) + '-' + str(counter) + '.png')
    #                                os.path.join('./samples', str(times), '.jpg'))
    #                    './{}/train_{:02d}_{:04d}.png'.format('./samples', epo, i))
#                        b = sample.eval({z:z_batch})
#                        b = b[10]
#                        b = (b + 1)*127.5
#                        cv2.imwrite(str(counter)+'.png', b)
                        show_mergeimage(samples, FLAGS.output_height, FLAGS.output_width, epo, counter, FLAGS.train)
                        print("samlpes: d_loss: %.8f, g_loss: %.8f" % (sample_d_loss, sample_g_loss))
            saver = tf.train.Saver()
            saver.save(sess, 'Model/model.ckpt')
    else:
        if os.path.exists('Model/model.ckpt.meta'):
            print('the generate image is in folder test_samples')
            with tf.Session(config=run_config) as sess: 
                saver = tf.train.Saver()
                saver.restore(sess, './Model/model.ckpt')
                z_batch = np.random.uniform(-1, 1, [FLAGS.batch_size, 100]).astype(np.float32)        
                samples = sess.run(sample, feed_dict = {z: z_batch}) 
                show_mergeimage(samples, FLAGS.output_height, FLAGS.output_width, 0, 0, FLAGS.train)
        else:
            raise Exception("[!] Train a model first")

if __name__ == '__main__':
  tf.app.run()             
            
        
        

    