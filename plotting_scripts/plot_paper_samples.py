#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:45:43 2021

@author: andrew
"""

import util as u
from neural_networks import unetpp
import numpy as np
import matplotlib.pyplot as plt
fields = ['ref','vel','wid']

#the outage case:
outage_size = 30
case_num = 62
buf_size=8
sz = u.SIZE['outage'][1]
mid = sz//2
win = [mid-outage_size,mid+outage_size]
truth = np.double(np.load('../data/outage_test_set.npy')[::5,:,:,:])
truth = truth[case_num,:,:,:]
samples = [truth]
inp = np.copy(truth)[np.newaxis,:,:,:]
mask = np.zeros((sz,sz))
mask[:,mid-outage_size:mid+outage_size] = 1.0
buf = np.linspace(1.0,0.0,buf_size+2)[1:-1]
mask[:,win[0]-buf_size:win[0]] = np.flip(buf[np.newaxis,:],axis=1)
mask[:,win[1]:win[1]+buf_size] = buf[np.newaxis,:]
mask = mask[np.newaxis,:,:,np.newaxis]
inp = np.concatenate((inp,mask),axis=-1)
l1_cnn = unetpp((*u.SIZE['outage'],4),base_channels=8,levels=7,growth=2)
l1_cnn.load_weights('../models/outage_l1/epoch_2000/variables/variables')
samples.append(l1_cnn.predict(inp).squeeze())
inp = np.concatenate((inp,np.random.normal(0,0.5,[1,sz,sz,1])),axis=3)
cgan_cnn = unetpp((*u.SIZE['outage'],5),base_channels=12,levels=7,growth=2)
cgan_cnn.load_weights('../models/outage_cgan/gen_epoch_0125000/variables/variables')
samples.append(cgan_cnn.predict(inp).squeeze())

FWID = 6
f = plt.figure(figsize=[FWID*3,FWID*2.75])
for j in range(3):
    for i in range(3):
        plt.subplot(3,3,i+1+j*3)
        u.plot_kazr_field(samples[j][:,:,i],fields[i])
        plt.plot(np.array([1,1])*win[0]*2/60,[0,256*30/1000],'k:')
        plt.plot(np.array([1,1])*win[1]*2/60,[0,256*30/1000],'k:')
        plt.text(0.2,7.1,'(' + chr(97+i+3*j)+')',fontsize=16)
#plt.savefig('../figures/paper/fig_03.png',dpi=600,bbox_inches='tight',pad_inches=0.1)
plt.savefig('../figures/paper/fig_03_lowres.png',dpi=100,bbox_inches='tight',pad_inches=0.1)



#the downfill case:
dsz = 37
case_num = 0
buf_size=8
sz = u.SIZE['downfill'][1]
mid = sz//2
truth = np.double(np.load('../data/downfill_test_set.npy')[::5,:,:,:])
truth = truth[case_num,:,:,:]
samples = [truth]
inp = np.copy(truth)[np.newaxis,:,:,:]
mask = np.zeros((sz,sz))
mask[:dsz,:] = 1.0
buf = np.linspace(1.0,0.0,buf_size+2)[1:-1]
mask[dsz:dsz+buf_size,:] = buf[:,np.newaxis]
mask = mask[np.newaxis,:,:,np.newaxis]
inp = np.concatenate((inp,mask),axis=-1)
l1_cnn = unetpp((*u.SIZE['downfill'],4),base_channels=8,levels=7,growth=2)
l1_cnn.load_weights('../models/downfill_l1/epoch_2000/variables/variables')
samples.append(l1_cnn.predict(inp).squeeze())
inp = np.concatenate((inp,np.random.normal(0,0.5,[1,sz,sz,1])),axis=3)
cgan_cnn = unetpp((*u.SIZE['downfill'],5),base_channels=12,levels=7,growth=2)
cgan_cnn.load_weights('../models/downfill_cgan/gen_epoch_0125000/variables/variables')
samples.append(cgan_cnn.predict(inp).squeeze())
FWID = 6
f = plt.figure(figsize=[FWID*3,FWID*2.75])
for j in range(3):
    for i in range(3):
        plt.subplot(3,3,i+1+j*3)
        u.plot_kazr_field(samples[j][:,:,i],fields[i])
        plt.plot([0,256*2/60],np.array([1,1])*dsz*30/1000,'k:')
        plt.text(0.2,7.1,'(' + chr(97+i+3*j)+')',fontsize=16)
#plt.savefig('../figures/paper/fig_05.png',dpi=600,bbox_inches='tight',pad_inches=0.1)
plt.savefig('../figures/paper/fig_05_lowres.png',dpi=100,bbox_inches='tight',pad_inches=0.1)

bsz = 6
truth = np.double(np.load('../data/blockage_test_set.npy')[::3,:,:,:])
N_buf = 8
sz = u.SIZE['blockage']
mid = sz[1]//2
win = [mid-bsz,mid+bsz]
bstart = 32
samples = [truth[150,:,:,:]]
inputs = samples[0][np.newaxis,:,:,:]

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
inputs = np.concatenate((inputs,mask),axis=3)
l1_cnn = unetpp((*u.SIZE['blockage'],4),base_channels=10,levels=6,growth=2)
l1_cnn.load_weights('../models/blockage_l1/epoch_2000/variables/variables')
samples.append(l1_cnn.predict(inputs).squeeze())
inputs = np.concatenate((inputs,np.random.normal(0,0.5,[1,sz[0],sz[1],1])),axis=3)
cgan_cnn = unetpp((*u.SIZE['blockage'],5),base_channels=12,levels=7,growth=2)
cgan_cnn.load_weights('../models/blockage_cgan/gen_epoch_0065000/variables/variables')
samples.append(cgan_cnn.predict(inputs).squeeze())
theta = (np.array(win)-19)*np.pi/180
r = np.array([bstart*100/1000,1024*100/1000])
FWID = 6
f = plt.figure(figsize=[FWID*3,FWID*2.75])
for j in range(3):
    for i in range(3):
        plt.subplot(3,3,i+1+j*3)
        u.plot_csapr_field(samples[j][:,19:-18,i],fields[i])
        plt.plot(r*np.cos(theta[0]),r*np.sin(theta[0]),'k:')
        plt.plot(r*np.cos(theta[1]),r*np.sin(theta[1]),'k:')
        plt.plot(r[0]*np.cos(theta),r[0]*np.sin(theta),'k:')
        plt.text(90,95,'(' + chr(97+i+3*j)+')',fontsize=16)
#plt.savefig('../figures/paper/fig_07.png',dpi=600,bbox_inches='tight',pad_inches=0.1)
plt.savefig('../figures/paper/fig_07_lowres.png',dpi=50,bbox_inches='tight',pad_inches=0.1)