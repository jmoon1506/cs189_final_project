import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *

class LatentAttention():
    def __init__(self):
        # self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.mnist = input_data.read_data_set_custom()
        self.n_samples = self.mnist.train.num_examples

        self.n_hidden = 500
        self.n_z = 20
        self.batchsize = 100
        self.step_size = 0.0001
        self.epoch = 0

        with tf.name_scope('Input'):
            self.images = tf.placeholder(tf.float32, [None, 16384])
            image_matrix = tf.reshape(self.images,[-1, 128, 128, 1])
            z_mean, z_stddev = self.recognition(image_matrix)
            samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
            guessed_z = z_mean + (z_stddev * samples)

        with tf.name_scope('Generator'):
            self.generated_images = self.generation(guessed_z)
            generated_flat = tf.reshape(self.generated_images, [self.batchsize, 128*128])

        with tf.name_scope('Loss'):
            self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)
            self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
            self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.step_size).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.name_scope('Recognizer'):
            with tf.variable_scope("Recognizer"):
                h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 128x128x1 -> 64x64x16
                h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 64x64x16 -> 32x32x32
                h3 = lrelu(conv2d(h2, 32, 64, "d_h3")) # 32x32x32 -> 16x16x64
                h4 = lrelu(conv2d(h3, 64, 128, "d_h4")) # 16x16x64 -> 8x8x128
                h4_flat = tf.reshape(h4,[self.batchsize,8*8*128])

                w_mean = dense(h4_flat, 8*8*128, self.n_z, "w_mean")
                w_stddev = dense(h4_flat, 8*8*128, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.name_scope('Generator'):
            with tf.variable_scope("Generator"):
                z_develop = dense(z, self.n_z, 8*8*128, scope='z_matrix')
                z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 8, 8, 128]))
                h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 16, 16, 64], "g_h1"))
                h2 = tf.nn.relu(conv_transpose(h1, [self.batchsize, 32, 32, 32], "g_h2"))
                h3 = tf.nn.relu(conv_transpose(h2, [self.batchsize, 64, 64, 16], "g_h3"))
                h4 = conv_transpose(h3, [self.batchsize, 128, 128, 1], "g_h4")
                h4 = tf.nn.sigmoid(h4)

        return h4

    def train(self):
        visualization = self.mnist.train.next_batch(self.batchsize)[0]
        visualization = self.mnist.train.next_batch(self.batchsize)[0]
        reshaped_vis = visualization.reshape(self.batchsize,128,128)
        ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        init = tf.global_variables_initializer()
        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.cost)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        with tf.Session() as  sess:
        #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(init)
            summary_writer = tf.summary.FileWriter("output/", graph=tf.get_default_graph())
            for epoch in range(self.epoch):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch = self.mnist.train.next_batch(self.batchsize)[0]
                    _, gen_loss, lat_loss, summary = sess.run([self.optimizer, self.generation_loss, self.latent_loss, merged_summary_op], feed_dict={self.images: batch})
                    summary_writer.add_summary(summary, epoch * self.batchsize + idx)
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples - 3) == 0:
                        print "epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss))
                        saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize,128,128)
                        ims("results/"+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))


model = LatentAttention()
model.train()
