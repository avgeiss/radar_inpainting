#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 08:38:18 2020

@author: andrew
"""

from keras.models import load_model
import util as u
from neural_networks import unetpp, blind_MAE
import netCDF4
from tensorflow.keras.utils import CustomObjectScope as cust
import numpy as np
from multiprocess import Pool
import baseline_schemes as schemes
from time import sleep

def inpaint(data,mask,func):
    data = np.copy(data)
    mask = np.copy(mask)
    #iterate through the three fields:
    for i in range(3):
        cdat = data[:,:,i]
        cdat = cdat*0.5+0.5
        cdat = func(cdat,np.copy(mask))
        cdat = cdat*2.0-1.0
        if i == 0:
            ref_mask = cdat<-0.5
            cdat[ref_mask] = -1.0
        if i == 1:
            cdat[ref_mask] = 0.0
        if i == 2:
            cdat[ref_mask] = -1.0
        data[:,:,i] = cdat
    return data

#the outage case:
def plot_outage_test_cases(outage_size=32):
    truth = np.double(np.load('../data/outage_test_set.npy')[::5,:,:,:])
    buf_size = 8
    sz = u.SIZE['outage'][1]
    mid = sz//2
    win = [mid-outage_size,mid+outage_size]
    
    #make the masked data:
    masked = np.copy(truth)
    masked[:,:,mid-outage_size:mid+outage_size,0] = -1.0
    masked[:,:,mid-outage_size:mid+outage_size,1] = 0.0
    masked[:,:,mid-outage_size:mid+outage_size,2] = -1.0
    
    #make a binary mask:
    mask = np.zeros((sz,sz))
    mask[:,mid-outage_size:mid+outage_size] = 1.0
    buf = np.linspace(1.0,0.0,buf_size+2)[1:-1]
    mask[:,win[0]-buf_size:win[0]] = np.flip(buf[np.newaxis,:],axis=1)
    mask[:,win[1]:win[1]+buf_size] = buf[np.newaxis,:]
    mask = mask[np.newaxis,:,:,np.newaxis]
    
    #get the L1_CNN results:
    inputs = np.concatenate((np.copy(masked),np.repeat(mask,masked.shape[0],axis=0)),axis=3)
    l1_cnn = unetpp((*u.SIZE['outage'],4),base_channels=8,levels=7,growth=2)
    l1_cnn.load_weights('../models/outage_l1/epoch_2000/variables/variables')
    l1 = l1_cnn.predict(inputs,verbose=1,batch_size=1)
    
    #get the cgan results:
    sz = l1.shape
    inputs = np.concatenate((inputs,np.random.normal(0,0.5,[sz[0],sz[1],sz[2],1])),axis=3)
    cgan_cnn = unetpp((*u.SIZE['outage'],5),base_channels=12,levels=7,growth=2)
    cgan_cnn.load_weights('../models/outage_cgan/gen_epoch_0125000/variables/variables')
    cgan = cgan_cnn.predict(inputs,verbose=1,batch_size=1)
    
    sz = u.SIZE['outage'][1]
    mask = np.zeros((sz,sz))
    mask[:,mid-outage_size:mid+outage_size] = 1.0
    def plot_samples(i):
        print(i,flush=True)
        u.plot(truth[i,:,:,:],'outage',fname='../figures/test_samples/outage/sample_' + str(i).zfill(4) + '_truth.png')
        u.plot(masked[i,:,:,:],'outage',fname='../figures/test_samples/outage/sample_' + str(i).zfill(4) + '_masked.png')
        u.plot(l1[i,:,:,:],'outage',fname='../figures/test_samples/outage/sample_' + str(i).zfill(4) + '_cnn.png')
        u.plot(cgan[i,:,:,:],'outage',fname='../figures/test_samples/outage/sample_' + str(i).zfill(4) + '_cgan.png')
        inpainted = inpaint(truth[i,:,:,:],mask,schemes.linear)
        u.plot(inpainted,'outage',fname='../figures/test_samples/outage/sample_' + str(i).zfill(4) + '_linear.png')
        inpainted = inpaint(truth[i,:,:,:],mask,schemes.laplace)
        u.plot(inpainted,'outage',fname='../figures/test_samples/outage/sample_' + str(i).zfill(4) + '_laplace.png')
        inpainted = inpaint(truth[i,:,:,:],mask,schemes.telea)
        u.plot(inpainted,'outage',fname='../figures/test_samples/outage/sample_' + str(i).zfill(4) + '_telea.png')
    p = Pool(10)
    p.map(plot_samples,range(truth.shape[0]))
    p.close()

# #the outage case:
def plot_downfill_test_cases(dsz=40):
    truth = np.double(np.load('./data/downfill_test_set.npy')[::4,:,:,:])
    buf_size = 8
    sz = u.SIZE['downfill'][1]
    
    #make the masked data:
    masked = np.copy(truth)
    masked[:,:dsz,:,0] = -1.0
    masked[:,:dsz,:,1] = 0.0
    masked[:,:dsz,:,2] = -1.0
    
    #make a binary mask:
    mask = np.zeros((sz,sz))
    mask[:dsz,:] = 1.0
    buf = np.linspace(1.0,0.0,buf_size+2)[1:-1]
    mask[dsz:dsz+buf_size,:] = buf[:,np.newaxis]
    mask = mask[np.newaxis,:,:,np.newaxis]
    
    #get the L1_CNN results:
    inputs = np.concatenate((np.copy(masked),np.repeat(mask,masked.shape[0],axis=0)),axis=3)
    l1_cnn = unetpp((*u.SIZE['downfill'],4),base_channels=8,levels=7,growth=2)
    l1_cnn.load_weights('../models/downfill_l1/epoch_2000/variables/variables')
    l1 = l1_cnn.predict(inputs,verbose=1,batch_size=1)
    
    #get the cgan results:
    sz = l1.shape
    inputs = np.concatenate((inputs,np.random.normal(0,0.5,[sz[0],sz[1],sz[2],1])),axis=3)
    cgan_cnn = unetpp((*u.SIZE['downfill'],5),base_channels=12,levels=7,growth=2)
    cgan_cnn.load_weights('../models/downfill_cgan/gen_epoch_0125000/variables/variables')
    cgan = cgan_cnn.predict(inputs,verbose=1,batch_size=1)
    
    sz = u.SIZE['downfill'][1]
    mask = np.zeros((sz,sz))
    mask[:dsz,:] = 1.0
    def plot_samples(i):
        print(i,flush=True)
        u.plot(truth[i,:,:,:],'outage',fname='../figures/test_samples/downfill/sample_' + str(i).zfill(4) + '_truth.png')
        u.plot(masked[i,:,:,:],'outage',fname='../figures/test_samples/downfill/sample_' + str(i).zfill(4) + '_masked.png')
        u.plot(l1[i,:,:,:],'outage',fname='../figures/test_samples/downfill/sample_' + str(i).zfill(4) + '_cnn.png')
        u.plot(cgan[i,:,:,:],'outage',fname='../figures/test_samples/downfill/sample_' + str(i).zfill(4) + '_cgan.png')
        inpainted = inpaint(truth[i,:,:,:],mask,schemes.marching_avg)
        u.plot(inpainted,'outage',fname='../figures/test_samples/downfill/sample_' + str(i).zfill(4) + '_marching_avg.png')
        inpainted = inpaint(truth[i,:,:,:],mask,schemes.repeat)
        u.plot(inpainted,'outage',fname='../figures/test_samples/downfill/sample_' + str(i).zfill(4) + '_repeat.png')
        inpainted = inpaint(truth[i,:,:,:],mask,schemes.efros)
        u.plot(inpainted,'outage',fname='../figures/test_samples/downfill/sample_' + str(i).zfill(4) + '_efros.png')
        
    p = Pool(4)
    p.map(plot_samples,range(truth.shape[0]))
    p.close()
    
def plot_blockage_test_cases(bsz=16):
    truth = np.double(np.load('../data/blockage_test_set.npy')[::25,:,:,:])
    N_buf = 8
    sz = u.SIZE['blockage'][1]
    mid = sz//2
    win = [mid-bsz,mid+bsz]
    bstart = 32
    
    #make the masked data:
    masked = np.copy(truth)
    masked[:,bstart:,mid-bsz:mid+bsz,0] = -1.0
    masked[:,bstart:,mid-bsz:mid+bsz,1] = 0.0
    masked[:,bstart:,mid-bsz:mid+bsz,2] = -1.0
    
    #make a binary mask:
    mask = np.zeros(u.SIZE['blockage'])
    mask[bstart:,win[0]:win[1]] = 1.0
    buf = np.linspace(1.0,0.0,N_buf+2)[1:-1]
    corner = np.flip(np.outer(buf,buf.T),axis=0)
    mask[bstart:,win[1]:win[1]+N_buf] = buf[np.newaxis,:]
    mask[bstart:,win[0]-N_buf:win[0]] = np.flip(buf[np.newaxis,:],axis=1)
    mask[bstart-N_buf:bstart,win[0]:win[1]] = np.flip(buf[:,np.newaxis],axis=0)
    mask[bstart-N_buf:bstart,win[1]:win[1]+N_buf] = corner
    mask[bstart-N_buf:bstart,win[0]-N_buf:win[0]] = np.flip(corner,axis=1)
    mask = mask[np.newaxis,:,:,np.newaxis]
    
    #get the L1_CNN results:
    inputs = np.concatenate((np.copy(masked),np.repeat(mask,masked.shape[0],axis=0)),axis=3)
    l1_cnn = load_model('../models/blockage_l1/epoch_2000',custom_objects={'blind_MAE': blind_MAE})
    l1 = l1_cnn.predict(inputs,verbose=1,batch_size=1)
    
    #get the cgan results:
    print('Checkpoint 2');sleep(1)
    sz = l1.shape
    inputs = np.concatenate((inputs,np.random.normal(0,0.5,[sz[0],sz[1],sz[2],1])),axis=3)
    cgan_cnn = load_model('../models/blockage_cgan/gen_epoch_0065000',custom_objects={'blind_MAE': blind_MAE})
    cgan = cgan_cnn.predict(inputs,verbose=1,batch_size=1)
    
    sz = u.SIZE['blockage'][1]
    mask = np.zeros(u.SIZE['blockage'])
    mask[bstart:,win[0]:win[1]] = 1.0
    def plot_samples(i):
        trm = [19,110]
        print(i,flush=True)
        u.plot(truth[i,:,trm[0]:trm[1],:],'blockage',fname='../figures/test_samples/blockage/sample_' + str(i).zfill(4) + '_truth.png')
        u.plot(masked[i,:,trm[0]:trm[1],:],'blockage',fname='../figures/test_samples/blockage/sample_' + str(i).zfill(4) + '_masked.png')
        u.plot(l1[i,:,trm[0]:trm[1],:],'blockage',fname='../figures/test_samples/blockage/sample_' + str(i).zfill(4) + '_cnn.png')
        u.plot(cgan[i,:,trm[0]:trm[1],:],'blockage',fname='../figures/test_samples/blockage/sample_' + str(i).zfill(4) + '_cgan.png')
        inpainted = inpaint(truth[i,:,:,:],mask,schemes.linear)[:,trm[0]:trm[1],:]
        u.plot(inpainted,'blockage',fname='../figures/test_samples/blockage/sample_' + str(i).zfill(4) + '_linear.png')
        inpainted = inpaint(truth[i,:,:,:],mask,schemes.laplace)[:,trm[0]:trm[1],:]
        u.plot(inpainted,'blockage',fname='../figures/test_samples/blockage/sample_' + str(i).zfill(4) + '_laplace.png')
        inpainted = inpaint(truth[i,:,:,:],mask,schemes.telea)[:,trm[0]:trm[1],:]
        u.plot(inpainted,'blockage',fname='../figures/test_samples/blockage/sample_' + str(i).zfill(4) + '_telea.png')
    p = Pool(24)
    p.map(plot_samples,range(truth.shape[0]))
    p.close()
    
    
# plot_outage_test_cases(outage_size=20)
plot_downfill_test_cases(dsz=40)
# plot_blockage_test_cases(bsz=8)
