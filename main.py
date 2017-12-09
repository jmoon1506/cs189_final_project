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
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 128x128x1 -> 64x64x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 64x64x16 -> 32x32x32
            h2_flat = tf.reshape(h2,[self.batchsize,32*32*32])

            w_mean = dense(h2_flat, 32*32*32, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, 32*32*32, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 32*32*32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 32, 32, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 64, 64, 16], "g_h1"))
            h2 = conv_transpose(h1, [self.batchsize, 128, 128, 1], "g_h2")
            h4 = tf.nn.sigmoid(h2)

        return h4

    def train(self):
        for i in range(5):
            visualization = self.mnist.test.next_batch(self.batchsize)[0]
            reshaped_vis = visualization.reshape(self.batchsize,128,128)
            base_dir = "all_classes_reconstruction_400_3_layers"+str(i)+ "/"
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            ims(base_dir+"base.jpg",merge(reshaped_vis[:64],[8,8]))
            # train
            saver = tf.train.Saver(max_to_keep=2)
            with tf.Session() as sess:
                generation_loss = []
                latent_loss = []
                sess.run(tf.initialize_all_variables())
                for epoch in range(30):
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
                gen_loss_file = open('gen_loss_0001_10x10_2layers.txt_'+str(self.n_z)+'latent.txt', 'w')
                lat_loss_file = open('lat_loss_0001_10x10_2layers.txt_'+str(self.n_z)+'latent.txt', 'w')

                for item in generation_loss:
                    gen_loss_file.write("%s\n" % item)
                for item in latent_loss:
                    lat_loss_file.write("%s\n" % item)

                # nx=ny=20
                # x_values = np.linspace(-3, 3, nx)
                # y_values = np.linspace(-3,3, ny)

                # canvas = np.empty((128*nx, 128*ny))
                # z_mu = np.random.randn(self.batchsize, self.n_z)
		# generated_test = sess.run(self.generated_images, feed_dict={self.guessed_z: z_mu})
		# generated_test = generated_test.reshape(self.batchsize,128,128)
		# ims("./sample_one/test.jpg",merge(generated_test[:100],[10,10]))

                # for i in range(100):
                #     z_mu = np.random.randn(self.batchsize, self.n_z)
                #     generated_test = sess.run(self.generated_images, feed_dict={self.guessed_z: z_mu})
                #     generated_test = generated_test.reshape(self.batchsize,128,128)
                #     ims("./3_classes_sample/sample_"+str(i)+".jpg",merge(generated_test[:25],[5,5]))
                
            return
        
model = LatentAttention()
model.train()
