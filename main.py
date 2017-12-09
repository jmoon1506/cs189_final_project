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

        self.n_z = 200
        self.batchsize = 100
        self.learning_rate = 0.0001
        self.filter_size = 10
        self.epochs = 30

        self.images = tf.placeholder(tf.float32, [None, 16384])
        image_matrix = tf.reshape(self.images,[-1, 128, 128, 1])
        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        self.guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(self.guessed_z)

        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 128*128])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1", self.filter_size, self.filter_size)) # 28x28x1 -> 14x14x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2", self.filter_size, self.filter_size)) # 14x14x16 -> 7x7x32
            h2_flat = tf.reshape(h2,[self.batchsize,32*32*32])

            w_mean = dense(h2_flat, 32*32*32, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, 32*32*32, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 32*32*32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 32, 32, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 64, 64, 16], "g_h1", self.filter_size, self.filter_size))
            h2 = conv_transpose(h1, [self.batchsize, 128, 128, 1], "g_h2", self.filter_size, self.filter_size)
            h2 = tf.nn.sigmoid(h2)

        return h2

    def train(self):
        with tf.Session() as sess:
            generation_loss = []
            latent_loss = []

            visualization = self.mnist.validation.next_batch(self.batchsize)[0]
            reshaped_vis = visualization.reshape(self.batchsize,128,128)
            base_dir = "results_processed/"
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            ims(base_dir+"base.jpg",merge(reshaped_vis[:64],[8,8]))
            # train
            saver = tf.train.Saver(max_to_keep=2)
            sess.run(tf.initialize_all_variables())
            for epoch in range(self.epochs):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch = self.mnist.train.next_batch(self.batchsize)[0]
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch})
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples - 3) == 0:
                        print "epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss))
                        generation_loss.append(np.mean(gen_loss))
                        latent_loss.append(np.mean(lat_loss))
                        saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize,128,128)
                        ims(base_dir+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))


            gen_loss_file = open('gen_loss.txt', 'w')
            lat_loss_file = open('lat_loss.txt', 'w')

            for item in generation_loss:
                gen_loss_file.write("%s\n" % item)
            for item in latent_loss:
                lat_loss_file.write("%s\n" % item)




model = LatentAttention()
model.train()
