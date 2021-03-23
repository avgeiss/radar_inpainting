#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:57:43 2021

@author: andrew
"""

#make_figure_02.py

import util as u
from neural_networks import unetpp
import numpy as np
import matplotlib.pyplot as plt
import baseline_schemes as schemes

#the outage case:
outage_size = 30
case_num = 62
buf_size=8
sz = u.SIZE['outage'][1]
mid = sz//2
win = [mid-outage_size,mid+outage_size]
truth = np.double(np.load('../data/outage_test_set.npy')[::5,:,:,0])
truth = truth[case_num,:,:]
FWID = 6
f = plt.figure(figsize=[FWID*4,FWID*1.85])


plt.subplot(2,4,1)
u.plot_kazr_field(np.copy(truth),'ref')
plt.plot(np.array([1,1])*win[0]*2/60,[0,256*30/1000],'k:')
plt.plot(np.array([1,1])*win[1]*2/60,[0,256*30/1000],'k:')
plt.title('a) Ground Truth Reflectivity (dBZ)')


inp = np.copy(truth)
mask = np.zeros((sz,sz))
mask[:,mid-outage_size:mid+outage_size] = 1.0
inp = inp*0.5+0.5
output = schemes.linear(inp,mask)
output = output*2.0-1.0
output[output<=-0.5] = -1.0
plt.subplot(2,4,2)
u.plot_kazr_field(output,'ref')
plt.plot(np.array([1,1])*win[0]*2/60,[0,256*30/1000],'k:')
plt.plot(np.array([1,1])*win[1]*2/60,[0,256*30/1000],'k:')
plt.title('b) Linear (dBZ)')


inp = np.copy(truth)
mask = np.zeros((sz,sz))
mask[:,mid-outage_size:mid+outage_size] = 1.0
inp = inp*0.5+0.5
output = schemes.laplace(inp,mask)
output = output*2.0-1.0
output[output<=-0.5] = -1.0
plt.subplot(2,4,3)
u.plot_kazr_field(output,'ref')
plt.plot(np.array([1,1])*win[0]*2/60,[0,256*30/1000],'k:')
plt.plot(np.array([1,1])*win[1]*2/60,[0,256*30/1000],'k:')
plt.title('c) Laplace (dBZ)')


inp = np.copy(truth)
mask = np.zeros((sz,sz))
mask[:,mid-outage_size:mid+outage_size] = 1.0
inp = inp*0.5+0.5
output = schemes.telea(inp,mask)
output = output*2.0-1.0
output[output<=-0.5] = -1.0
plt.subplot(2,4,4)
u.plot_kazr_field(output,'ref')
plt.plot(np.array([1,1])*win[0]*2/60,[0,256*30/1000],'k:')
plt.plot(np.array([1,1])*win[1]*2/60,[0,256*30/1000],'k:')
plt.title('d) Telea (dBZ)')


#the downfill case:
dsz = 37
case_num = 0
buf_size=8
sz = u.SIZE['downfill'][1]
mid = sz//2
truth = np.double(np.load('../data/downfill_test_set.npy')[::5,:,:,:])
truth = truth[case_num,:,:,0]

plt.subplot(2,4,5)
u.plot_kazr_field(np.copy(truth),'ref')
plt.plot([0,256*2/60],np.array([1,1])*dsz*30/1000,'k:')
plt.title('e) Ground Truth Reflectivity (dBZ)')


inp = np.copy(truth)
mask = np.zeros((sz,sz))
mask[:dsz,:] = 1.0
inp = inp*0.5+0.5
output = schemes.repeat(inp,mask)
output = output*2.0-1.0
output[output<=-0.5] = -1.0
plt.subplot(2,4,6)
u.plot_kazr_field(output,'ref')
plt.plot([0,256*2/60],np.array([1,1])*dsz*30/1000,'k:')
plt.title('f) Repeat (dBZ)')


inp = np.copy(truth)
mask = np.zeros((sz,sz))
mask[:dsz,:] = 1.0
inp = inp*0.5+0.5
output = schemes.marching_avg(inp,mask)
output = output*2.0-1.0
output[output<=-0.5] = -1.0
plt.subplot(2,4,7)
u.plot_kazr_field(output,'ref')
plt.plot([0,256*2/60],np.array([1,1])*dsz*30/1000,'k:')
plt.title('g) Marching AVG. (dBZ)')


inp = np.copy(truth)
mask = np.zeros((sz,sz))
mask[:dsz,:] = 1.0
inp = inp*0.5+0.5
output = schemes.efros(inp,mask)
output = output*2.0-1.0
output[output<=-0.5] = -1.0
plt.subplot(2,4,8)
u.plot_kazr_field(output,'ref')
plt.plot([0,256*2/60],np.array([1,1])*dsz*30/1000,'k:')
plt.title('h) Efros (dBZ)')

plt.savefig('../figures/paper/fig_02.png',dpi=600,bbox_inches='tight',pad_inches=0.1,pil_kwargs={'quality': 30})