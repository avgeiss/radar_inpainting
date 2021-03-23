#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:30:26 2020

@author: andrew
"""
#imports here:
import tensorflow as tf
from keras.layers import Conv2D, Input, UpSampling2D, MaxPooling2D, Dense
from keras.layers import LeakyReLU, Lambda, concatenate, Dropout, Flatten
import keras.backend as K
from keras.models import Model
from keras.utils import plot_model

#custom loss functions
def blind_MSE(y_true,y_pred):
    #this is a custom version of MAE that weights based on the filter provided to the CNN
    mse = (y_true[:,:,:,:3]-y_pred)**2
    filt = y_true[:,:,:,3]
    filt = tf.stack([filt,filt,filt],axis=3)
    weighted_mse = tf.reduce_mean(mse*filt)/tf.reduce_mean(filt)
    return weighted_mse

def blind_MAE(y_true,y_pred):
    #this is a custom version of MAE that weights based on the filter provided to the CNN
    mae = tf.math.abs(y_true[:,:,:,:3]-y_pred)
    filt = y_true[:,:,:,3]
    filt = tf.stack([filt,filt,filt],axis=3)
    weighted_mae = tf.reduce_mean(mae*filt)/tf.reduce_mean(filt)
    return weighted_mae

#combines the known input data with the generated data for the unknown region
def merge_output(xin,x):
    filt = xin[:,:,:,3]
    nchan = 3
    outputs = []
    for i in range(nchan):
        merged = filt*x[:,:,:,i] + (1.0-filt)*xin[:,:,:,i]
        outputs.append(K.expand_dims(merged))
    return Lambda(lambda x: x)(concatenate(outputs))

def preprocess(x):
    #blank out the inputs where the filter is:
    filt = tf.floor(x[:,:,:,3])
    x0 = x[:,:,:,0]*(1-filt) - filt    #set missing ref to -1
    x1 = x[:,:,:,1]*(1-filt)           #set missing vel to 0
    x2 = x[:,:,:,2]*(1-filt) - filt    #set missing wid to -1
    x = tf.stack([x0,x1,x2,x[:,:,:,3]],axis=3)
    return Lambda(lambda x: x)(x)

#defines the convolutions done in the unet
def conv(x,channels,filter_size=3):
    x = Conv2D(channels,(filter_size,filter_size),padding='same',activation='linear')(x)
    x = LeakyReLU(0.2)(x)
    return x

#creates unet ++ style network
def unetpp(INPUT_SIZE,base_channels=8,levels=7,growth=2):
    #defines the unet++
    xin = Input(INPUT_SIZE)
    xpre = preprocess(xin)
    net = []
    for lev in range(levels):
        if lev == 0:
            net.append([concatenate([xpre,conv(xpre,base_channels)])])
        else:
            net_layer = []
            for proc in range(lev+1):
                inputs = []
                if proc < lev:
                    inputs.append(MaxPooling2D((2,2))(net[lev-1][proc]))
                if proc > 0:
                    inputs.append(UpSampling2D((2,2))(net_layer[proc-1]))
                    inputs.append(net[lev-1][proc-1])
                if len(inputs)>1:
                    inputs = concatenate(inputs)
                else:
                    inputs = inputs[0]
                output = conv(inputs,int(base_channels*growth**(lev-proc)))
                if proc>0:
                    output = concatenate([output,net[lev-1][proc-1]])
                net_layer.append(output)
            net.append(net_layer)
    x = conv(net[-1][-1],base_channels*levels)
    #combine input with generated output:
    xout = Conv2D(3,(1,1),activation='tanh',padding='same')(x)
    xout = merge_output(xin,xout)
    cnn = Model(xin,xout)
    return cnn

# #create a discriminator network:
def discriminator(in_shape,base_channels=18,growth=2):
    def disc_block(xin,channels):
        tensors = [xin]
        for i in range(2):
            if i == 0:
                x = tensors[0]
            else:
                x = concatenate(tensors)
            x = conv(x,channels)
            tensors.append(x)
        x = conv(concatenate(tensors),channels)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.1)(x)
        return x
    
    xin = Input(in_shape)
    x = xin
    channels = base_channels
    for i in range(7):
        x = disc_block(x,int(channels*growth**i))
    x = Flatten()(x)
    xout = Dense(1,activation='sigmoid',name='classification_layer')(x)
    return Model(xin, xout)

def gan(in_shape,gen,dis):
    dis.trainable = False
    xin = Input(in_shape)
    generated = gen(xin)
    classification = dis(concatenate([generated,K.expand_dims(Lambda(lambda x: x[:,:,:,3])(xin))]))
    return Model(inputs = xin, outputs = [generated, classification])

# from keras.utils import plot_model
# cnn = unetpp((256,256,4),base_channels=8,levels=7,growth=2)
# plot_model(cnn,'./figures/cnn_plots/l1_outage.png',show_shapes=True)
# cnn = unetpp((1024,128,4),base_channels=10,levels=6,growth=2)
# plot_model(cnn,'./figures/cnn_plots/l1_blockage.png',show_shapes=True)
# cnn = unetpp((256,256,4),base_channels=12,levels=7,growth=2)
# plot_model(cnn,'./figures/cnn_plots/cgan_outage.png',show_shapes=True)
# cnn = unetpp((1024,128,4),base_channels=14,levels=6,growth=2)
# plot_model(cnn,'./figures/cnn_plots/cgan_blockage.png',show_shapes=True)
# cnn = discriminator((256,256,4),base_channels=12,growth=1.75)
# plot_model(cnn,'./figures/cnn_plots/discriminator_outage.png',show_shapes=True)
# cnn = discriminator((1024,128,4),base_channels=14,growth=1.75)
# plot_model(cnn,'./figures/cnn_plots/discriminator_blockage.png',show_shapes=True)