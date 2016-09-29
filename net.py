from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import argparse
import numpy as np
from scipy.stats import norm
import tensorflow as tf

def conv(input,name,in_channel,out_channel,subsample):
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,in_channel,out_channel],
                            dtype=tf.float32,stddev = 1e-1),
                            name= 'weights')
        conv = tf.nn.con2d(input,kernel,subsample,padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[out_channel],dtyp=tf.float32),
                                trainable=True,
                                name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name=scope)
    return conv1 
    
def deconv_unativated(input,name,in_channel,out_channel,subsample):
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,in_channel,out_channel],
                            dtype=tf.float32,stddev=1e-1),
                            name = 'weights')
        deconv = tf.nn.con2d_transpose(input,
                                        kernel,
                                        [input.shape[0],
                                        input.shape[1]*subsample[0],
                                        input.shape[2]*subsample[2],
                                        input.shape[3].subsample[3]],
                                        strides=subsample,
                                        padding = 'SAME')
        
    return deconv
    
def conv(input,name,in_channel,out_channel,subsample):
    x = deconv_unativated(input,name,in_channel,out_channel,subsample)
    out = tf.nn.relu(x)
    return out
    
def generator(X):
    #conv1
    conv1 = conv(X,'conv1',1,256,[1,4,4,])
    conv2 = conv(conv1,'conv2',256,1024,[1,4,4,1])
    deconv1 = deconv(conv2,'deconv1',1024,512,[1,2,2,1])
    deconv2 = deconv(deconv1,'deconv2',512,256,[1,2,2,1])
    deconv3 = deconv(deconv2,'deconv3',256,256,[1,2,2,1])
    deconv4 = deconv(deconv3,'deconv4',256,128,[1,2,2,1])
    deconv5 = deconv_unativated(deconv4,'deconv5',128,2,[1,1,1,1])
    gX = tf.nn.tanh(deconv5)
    return gX
    
def discriminator(X):
    conv3 = conv(X,'conv3',2,128,[1,2,2,1])
    conv4 = conv(conv3,'conv4',128,256,[1,2,2,1])
    conv5 = conv{conv4,'conv5',256,512,[1,2,2,1]}
    conv6 = conv(conv5,'conv6',512,1024,[1,2,2,1])
    with tf.name_scope('flatten') as scope:
        conv_flated = tf.reshape(conv6,[-1,1024*10*8])
        wy = tf.Variable(tf.truncated_normal([1024*10*8,1],dtype=tf.float32,stddev=1e-1),name = 'weights')
        y = tf.nn.sigmoid(tf.matmul(conv_flated,wy))
    return y

def optimizer(loss, var_list):
    initial_learning_rate = 0.005
    decay = 0.95
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer
    

class GAN():
    def __init__(self,
                shape , #high,width,channel
                l2 = 1e-5,
                nvis = 196,
                b1 = 0.5,
                nc = 2,
                nbatch = 128,
                npx = 64,
                batch_size = 128,
                niter = 1000,
                lr = 0.0002,
                ntrain = 25000)
        self.l2 = l2
        self.nvis = nvis
        self.b1 = b1
        self.nc = nc
        self.nbatch = nbatch
        self.npx = npx 
        self.batch_size = batch_size
        self.nx = npx*npx*nc
        self.niter = niter
        self.lr = lr
        self.ntrain = ntrain
        
        self._create_model()
        
    def _create_model():
        with tf.variable_scope('G'):
            self.X = tf.placeholder(tf.float32,shape =(self.shape[0],
                                                       self.shape[1],
                                                       1,
                                                       self.batch_size))
            self.gX = generator(self.X)
            
        with tf.variable_scope('D'):
            self.target = tf.placeholder(tf.float32,shape=(self.shape[0],
                                                           self.shape[1],
                                                           2,
                                                           self.batch_size))
                                                           
            self.Y = discriminator(self.target)
            scope.reuse_variable()
            self.gY = discriminator(self.gX)
            
        self.loss_d = tf.reduce_mean(-tf.log(self.Y) - tf.log(1 - self.gY))
        self.loss_g = tf.reduce_mean(-tf.log(self.gY))   
        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]    
        self.opt_d = optimizer(self.loss_d,self.d_params)
        self.opt_g = optimizer(self.loss_g,self.g_params)
        
    def train():
        with tf.Session() as session:
            tf.initialize_all_variable().run()
            
            for step in xrange(self.iter):
                x = load_x()
                target = load_t()
                loss_d,_ = session.run([self.loss_d,self.opt_d],{ \
                    self.X:np.reshape(x,(self.shape[0],self.shape[1],1,self.batch_size)), \
                    self.target : np.reshape(target,(self.shape[0],self.shape[1],2,self.batch_size)) \
                    })
            
                x = load_x()
                loss_g,_ = session.run([self.loss_g,self.opt_g],{ \
                    self.X: np.reshape(x,(self.shape[0],self.shape[1],1,self.batch_size)) \
                    })
                
            
    
    def sample():