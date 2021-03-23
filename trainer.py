#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 09:57:16 2020

@author: andrew
"""


import util as u
import neural_networks as nets
import numpy as np
import keras.backend as K
from keras.callbacks import History
from keras.optimizers import Adam
import gc


def train_l1(case):
    #define constants
    BATCH_SIZE = 8
    EPOCH_SIZE = 250*BATCH_SIZE
    N_EPOCHS = 2000
    LR_SCHEDULE = [1600,1900]

    if case == 'blockage':
        base_channels=10
        levels=6
    else:
        base_channels=8
        levels=7
        
    #prep the cnn:
    history = History()
    cnn = nets.unetpp((*u.SIZE[case],4),base_channels=base_channels,levels=levels,growth=2)
    cnn.summary()
    cnn.compile(optimizer=Adam(learning_rate=0.0005),loss=nets.blind_MAE)
    batch = u.BATCH_FUNC[case]
    
    #load the data:
    data = np.load(u.data_dir + u.INST[case] + '.npy')
    x_samples = np.load(u.data_dir  + case + '_samples.npy')
    
    #prep training batch:
    x = np.zeros((EPOCH_SIZE,*u.SIZE[case],4),dtype='float16')
    
    #the training loop:
    for epoch in range(N_EPOCHS):
        gc.collect()
        print('EPOCH ' + str(epoch) + ':')
        
        #check if the learning rate should be reduced:
        if epoch in LR_SCHEDULE:
            print('Reducing Learning Rate...')
            K.set_value(cnn.optimizer.lr,K.get_value(cnn.optimizer.lr)*0.1)
            
        #get a new epoch:
        batch(x,data)
        
        #train for an epoch:
        cnn.fit(x,x,batch_size=BATCH_SIZE,verbose=1,callbacks=[history])
        
        #save and plot the loss
        np.save('./models/' + case + '_l1/loss.npy',np.array(history.history['loss']))
        
        #save the model
        if epoch%100 == 99:
            cnn.save('./models/' + case + '_l1/epoch_' + str(epoch+1).zfill(4))
            
            #plot sample outputs
            outputs = cnn.predict(x_samples,batch_size=BATCH_SIZE)
            r,v,w = outputs[:,:,:,0], outputs[:,:,:,1], outputs[:,:,:,2]
            v[r<-0.5] = 0.0
            w[r<-0.5] = -1.0
            r[r<-0.5] = -1.0
            outputs[:,:,:,0], outputs[:,:,:,1], outputs[:,:,:,2] = r, v, w
            for i in range(outputs.shape[0]):
                u.plot(outputs[i,:,:,:3],case,fname='./figures/training_samples/' + case + '_l1/sample' + str(i) + '_epoch' + str(epoch+1).zfill(4) + '.png')
            gc.collect()
            

def train_cgan(case):
    #define constants
    BATCH_SIZE = 16
    N_BATCH = 200_000
    LR = 0.000001
    START_EPOCH = 125_000
    TEST_FREQ = 1_000
    SAVE_FREQ = 1_000

    if case == 'blockage':
        base_channels=14
        levels=6
    else:
        base_channels=12
        levels=7
        
    #prep the cnn:
    gen = nets.unetpp((*u.SIZE[case],5),base_channels=base_channels,levels=levels,growth=2)
    dis = nets.discriminator((*u.SIZE[case],4),base_channels=base_channels,growth=1.75)
    losses = []
    if START_EPOCH>0:
        gen.load_weights('./models/outage_cgan/gen_epoch_' + str(START_EPOCH).zfill(7) + '/variables/variables')
        dis.load_weights('./models/outage_cgan/dis_epoch_' + str(START_EPOCH).zfill(7) + '/variables/variables')
        losses = np.load('./models/outage_cgan/loss.npy')
        losses = list(losses[:START_EPOCH,:])
        
    dis.compile(optimizer=Adam(learning_rate=LR*1.5,beta_1=0.5),loss='binary_crossentropy')
    gan = nets.gan((*u.SIZE[case],5),gen,dis)
    gan.compile(optimizer=Adam(learning_rate=LR,beta_1=0.5),loss=[nets.blind_MAE, 'binary_crossentropy'],loss_weights=[8.0,1.0])
    gen.summary()
    dis.summary()
    
    #get the data:
    data = np.load(u.data_dir + u.INST[case] + '.npy')
    x_samples = np.load(u.data_dir  + case + '_samples.npy')
    sample_noise = np.random.normal(0.0,scale=0.5,size=(*x_samples[:,:,:,0].shape,1))
    x_samples = np.concatenate((x_samples,sample_noise),axis=3)
    
    #training loop
    gbatch = np.zeros((BATCH_SIZE,*u.SIZE[case],5),dtype='float16')
    dbatch = np.zeros((BATCH_SIZE,*u.SIZE[case],4),dtype='float16')
    for batch_num in range(START_EPOCH,N_BATCH):
        
        #training step:
        glabels = u.gen_batch(u.BATCH_FUNC[case],gbatch,data)
        dlabels = u.dis_batch(u.BATCH_FUNC[case],dbatch,data,gen)
        gloss = gan.train_on_batch(gbatch,[gbatch,glabels])
        print(type(gloss))
        dloss = dis.train_on_batch(dbatch,dlabels)
        losses.append(gloss+[dloss])
        cur_loss = np.round(np.mean(losses[-1000:],axis=0),5)
        print('BATCH ' + str(batch_num).zfill(7) + ': ' + str(cur_loss))
        
        #if the generator is doing significantly worse give it some more training iterations:
        for i in range(int(np.floor(cur_loss[2]/cur_loss[3]))):
            glabels = u.gen_batch(u.BATCH_FUNC[case],gbatch,data)
            gan.train_on_batch(gbatch,[gbatch,glabels])
            print('.',end='')
            
        #save test samples
        if batch_num % TEST_FREQ == TEST_FREQ-1:
            outputs = gen.predict(x_samples)
            r,v,w = outputs[:,:,:,0], outputs[:,:,:,1], outputs[:,:,:,2]
            v[r<-0.5] = 0.0
            w[r<-0.5] = -1.0
            r[r<-0.5] = -1.0
            outputs[:,:,:,0], outputs[:,:,:,1], outputs[:,:,:,2] = r, v, w
            for i in range(outputs.shape[0]):
                u.plot(outputs[i,:,:,:3],case,fname='./figures/training_samples/' + case + '_cgan/sample' + str(i) + '_epoch' + str(batch_num+1).zfill(7) + '.png')
        
        #save the cnns
        if batch_num % SAVE_FREQ == SAVE_FREQ-1:
            gen.save('./models/' + case + '_cgan/gen_epoch_' + str(batch_num+1).zfill(7))
            dis.save('./models/' + case + '_cgan/dis_epoch_' + str(batch_num+1).zfill(7))
        
        #save loss and garbage collect
        if batch_num % 1000 == 999:
            gc.collect()
            np.save('./models/' + case + '_cgan/loss.npy',np.array(losses))  