#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:02:23 2020

@author: andrew
"""

import numpy as np
import util as u
import baseline_schemes as schemes
from multiprocess import Pool
from neural_networks import unetpp
import gc

def emd(x,y,inst='kazr'):
    if inst == 'kazr':
        rng = [[-10,40],[-12,12],[0,5]]
    elif inst=='csapr':
        rng = [[0,60],[-33,33],[0,5.5]]
    EMD = []
    norm = x.shape[1]*x.shape[2]
    nbin = 64
    for i in range(x.shape[0]):
        femd = []
        for j in range(x.shape[-1]):
            hy = np.histogram(y[i,:,:,j],bins=nbin,range=rng[j])[0]
            hx = np.histogram(x[i,:,:,j],bins=nbin,range=rng[j])[0]
            cy = np.cumsum(hy)/norm
            cx = np.cumsum(hx)/norm
            femd.append(100*np.abs(np.sum(cx-cy))/nbin)
        EMD.append(femd)
    return np.mean(np.array(EMD),axis=0)
    
def psd(x):
    #trim the input to a power of two:
    x = x[:,:int(2**np.floor(np.log2(x.shape[1]))),...]
    #do fourier transform
    fft = np.abs(np.fft.rfft(x,axis=1))
    mnfft = np.nanmean(fft,axis=2)
    mnpsd = 10*np.log10(mnfft**2)
    mnpsd[np.abs(mnpsd)>10000] = np.nan
    mnpsd = np.nanmean(mnpsd,axis=0)
    return mnpsd[1:-1]

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


#compute metrics for outage case##############################################
def eval_outage_errors(outage_size):
    
    test_data = list(np.double(np.load('./data/outage_test_set.npy')))
    buf_size = 8
    sz = u.SIZE['outage'][1]
    mid = sz//2
    win = [mid-outage_size,mid+outage_size]
    
    #get the ground truth:
    truth = []
    for sample in test_data:
        truth.append(sample[:,mid-outage_size:mid+outage_size,:])
    truth = np.array(truth)
    
    #make a binary mask:
    mask = np.zeros((sz,sz))
    mask[:,mid-outage_size:mid+outage_size] = 1.0
    
    #compute the baseline schemes:
    p = Pool(11)
    linear = p.map(lambda x: inpaint(x,mask,schemes.linear),test_data)
    linear = np.array(linear)[:,:,win[0]:win[1],:]
    laplace = p.map(lambda x: inpaint(x,mask,schemes.laplace),test_data)
    laplace = np.array(laplace)[:,:,win[0]:win[1],:]
    telea = p.map(lambda x: inpaint(x,mask,schemes.telea),test_data)
    telea = np.array(telea)[:,:,win[0]:win[1],:]
    p.close()
    
    #prep the inputs for the CNNs:
    buf = np.linspace(1.0,0.0,buf_size+2)[1:-1]
    mask[:,win[0]-buf_size:win[0]] = np.flip(buf[np.newaxis,:],axis=1)
    mask[:,win[1]:win[1]+buf_size] = buf[np.newaxis,:]
    mask = mask[:,:,np.newaxis]
    for i in range(len(test_data)):
        sample = test_data[i]
        sample[:,win[0]:win[1],0] = -1.0
        sample[:,win[0]:win[1],1] = 0.0
        sample[:,win[0]:win[1],2] = -1.0
        test_data[i] = np.concatenate((sample,mask,np.random.normal(0,0.5,mask.shape)),axis=2)
    
    #the l1 case
    l1_cnn = unetpp((*u.SIZE['outage'],4),base_channels=8,levels=7,growth=2)
    l1_cnn.load_weights('./models/outage_l1/epoch_2000/variables/variables')
    l1 = l1_cnn.predict(np.array(test_data)[:,:,:,:4],verbose=1,batch_size=1)
    l1 = l1[:,:,win[0]:win[1],:]
    ref_mask = l1[:,:,:,0]<-0.5
    for i in range(3):
        if i == 0 or i == 2:
            l1[:,:,:,i][ref_mask] = -1.0
        else:
            l1[:,:,:,i][ref_mask] = 0.0
    del l1_cnn;gc.collect()
            
    #the cgan case
    cgan_cnn = unetpp((*u.SIZE['outage'],5),base_channels=12,levels=7,growth=2)
    cgan_cnn.load_weights('./models/outage_cgan/gen_epoch_0150000/variables/variables')
    cgan = cgan_cnn.predict(np.array(test_data),verbose=1,batch_size=1)[:,:,win[0]:win[1],:]
    ref_mask = cgan[:,:,:,0]<-0.5
    for i in range(3):
        if i == 0 or i == 2:
            cgan[:,:,:,i][ref_mask] = -1.0
        else:
            cgan[:,:,:,i][ref_mask] = 0.0
    del cgan_cnn;gc.collect()
    
    #dimensionalize and collect the results:
    def dim(x):
        fields = ['ref','vel','wid']
        for i in range(3):
            x[:,:,:,i] = u.inv_standardize(x[:,:,:,i],fields[i],'kazr')
        return x
    infilled = [dim(linear),dim(laplace),dim(telea),dim(l1),dim(cgan)]
    truth = dim(truth)
    
    #compute all of the error metrics:
    MAE, MSE, PSD, PSDt, EMD = [], [], [], [], []
    for inp in infilled:
        MSE.append(np.mean((truth-inp)**2,axis = (0,1,2)))
        MAE.append(np.mean(np.abs(truth-inp),axis = (0,1,2)))
        EMD.append(emd(inp,truth,inst='kazr'))
        PSD.append(psd(inp))
        PSDt.append(psd(inp.transpose((0,2,1,3))))
    
    true_psd = psd(truth)
    true_psdt = psd(truth.transpose((0,2,1,3)))
    
    return MAE, MSE, EMD, PSD, true_psd, PSDt, true_psdt

#precompute efros scheme because it takes a long time:
def precomp_efros(dfsz):
    print('Computing Efros & Leung Inpainting for Blind-zone Size: ' + str(dfsz))
    test_data = list(np.double(np.load('./data/downfill_test_set.npy')))
    sz = u.SIZE['downfill'][1]
    mask = np.zeros((sz,sz))
    mask[:dfsz,:] = 1.0
    def efros_helper(idx):
        print(idx,end=',',flush=True)
        return schemes.efros3(test_data[idx],mask)
    p = Pool(4)
    output = p.map(efros_helper,np.arange(len(test_data)),chunksize=1)
    p.close()
    np.save('./data/downfill_efros_' + str(dfsz) + '.npy',output)


def eval_downfill_errors(dsz):
    
    test_data = list(np.double(np.load('./data/downfill_test_set.npy')))
    buf_size = 8
    sz = u.SIZE['downfill'][1]
    
    #get the ground truth:
    truth = []
    for sample in test_data:
        truth.append(sample[:dsz,:,:])
    truth = np.array(truth)
    
    #make a binary mask:
    mask = np.zeros((sz,sz))
    mask[:dsz,:] = 1.0
    
    #compute the baseline schemes:
    p = Pool(11)
    marching_avg = p.map(lambda x: inpaint(x,mask,schemes.marching_avg),test_data)
    marching_avg = np.array(marching_avg)[:,:dsz,:,:]
    repeat = p.map(lambda x: inpaint(x,mask,schemes.repeat),test_data)
    repeat = np.array(repeat)[:,:dsz,:,:]
    p.close()
    efros = np.load('./data/downfill_efros_' + str(dsz) + '.npy')[:len(test_data),:,:,:]
    efros = efros[:,:dsz,:,:]
    
    #prep the inputs for the CNNs:
    buf = np.linspace(1.0,0.0,buf_size+2)[1:-1]
    mask[dsz:dsz+buf_size,:] = buf[:,np.newaxis]
    mask = mask[:,:,np.newaxis]
    for i in range(len(test_data)):
        sample = test_data[i]
        sample[:dsz,:,0] = -1.0
        sample[:dsz,:,1] = 0.0
        sample[:dsz,:,2] = -1.0
        test_data[i] = np.concatenate((sample,mask,np.random.normal(0,0.5,mask.shape)),axis=2)
    
    #the l1 case
    l1_cnn = unetpp((*u.SIZE['downfill'],4),base_channels=8,levels=7,growth=2)
    l1_cnn.load_weights('./models/downfill_l1/epoch_2000/variables/variables')
    l1 = l1_cnn.predict(np.array(test_data)[:,:,:,:4],verbose=1,batch_size=1)
    l1 = l1[:,:dsz,:,:]
    ref_mask = l1[:,:,:,0]<-0.5
    for i in range(3):
        if i == 0 or i == 2:
            l1[:,:,:,i][ref_mask] = -1.0
        else:
            l1[:,:,:,i][ref_mask] = 0.0
    del l1_cnn;gc.collect()
            
    #the cgan case
    cgan_cnn = unetpp((*u.SIZE['outage'],5),base_channels=12,levels=7,growth=2)
    cgan_cnn.load_weights('./models/downfill_cgan/gen_epoch_0125000/variables/variables')
    cgan = cgan_cnn.predict(np.array(test_data),verbose=1,batch_size=1)[:,:dsz,:,:]
    ref_mask = cgan[:,:,:,0]<-0.5
    for i in range(3):
        if i == 0 or i == 2:
            cgan[:,:,:,i][ref_mask] = -1.0
        else:
            cgan[:,:,:,i][ref_mask] = 0.0
    del cgan_cnn;gc.collect()
    
    #dimensionalize and collect the results:
    def dim(x):
        fields = ['ref','vel','wid']
        for i in range(3):
            x[:,:,:,i] = u.inv_standardize(x[:,:,:,i],fields[i],'kazr')
        return x
    infilled = [dim(marching_avg),dim(repeat),dim(efros),dim(l1),dim(cgan)]
    truth = dim(truth)
    
    #compute all of the error metrics:
    MAE, MSE, PSD, PSDt, EMD = [], [], [], [], []
    for inp in infilled:
        MSE.append(np.mean((truth-inp)**2,axis = (0,1,2)))
        MAE.append(np.mean(np.abs(truth-inp),axis = (0,1,2)))
        EMD.append(emd(inp,truth))
        PSD.append(psd(inp))
        PSDt.append(psd(inp.transpose((0,2,1,3))))
    
    true_psd = psd(truth)
    true_psdt = psd(truth.transpose((0,2,1,3)))
    
    return MAE, MSE, EMD, PSD, true_psd, PSDt, true_psdt

def eval_blockage_errors(bsz):
    test_data = list(np.double(np.load('./data/blockage_test_set.npy')))
    N_buf = 8
    bstart = 32
    sz = u.SIZE['blockage'][1]
    mid = sz//2
    win = [mid-bsz,mid+bsz]
    
    #get the ground truth:
    truth = []
    for sample in test_data:
        truth.append(sample[:,win[0]:win[1],:])
    truth = np.array(truth)
    
    #make a binary mask:
    mask = np.zeros(u.SIZE['blockage'])
    mask[bstart:,win[0]:win[1]] = 1.0
    
    #compute the baseline schemes:
    p = Pool(22)
    linear = p.map(lambda x: inpaint(x,mask,schemes.linear),test_data)
    linear = np.array(linear)[:,:,win[0]:win[1],:]
    laplace = p.map(lambda x: inpaint(x,mask,schemes.laplace),test_data)
    laplace = np.array(laplace)[:,:,win[0]:win[1],:]
    telea = p.map(lambda x: inpaint(x,mask,schemes.telea),test_data)
    telea = np.array(telea)[:,:,win[0]:win[1],:]
    p.close()
    
    #prep the inputs for the CNNs:
    buf = np.linspace(1.0,0.0,N_buf+2)[1:-1]
    corner = np.flip(np.outer(buf,buf.T),axis=0)
    mask[bstart:,win[1]:win[1]+N_buf] = buf[np.newaxis,:]
    mask[bstart:,win[0]-N_buf:win[0]] = np.flip(buf[np.newaxis,:],axis=1)
    mask[bstart-N_buf:bstart,win[0]:win[1]] = np.flip(buf[:,np.newaxis],axis=0)
    mask[bstart-N_buf:bstart,win[1]:win[1]+N_buf] = corner
    mask[bstart-N_buf:bstart,win[0]-N_buf:win[0]] = np.flip(corner,axis=1)
    mask = mask[:,:,np.newaxis]
    for i in range(len(test_data)):
        sample = test_data[i]
        sample[bstart:,win[0]:win[1],0] = -1.0
        sample[bstart:,win[0]:win[1],1] = 0.0
        sample[bstart:,win[0]:win[1],2] = -1.0
        test_data[i] = np.concatenate((sample,mask,np.random.normal(0,0.5,mask.shape)),axis=2)
    
    #the l1 case
    l1_cnn = unetpp((*u.SIZE['blockage'],4),base_channels=10,levels=6,growth=2)
    l1_cnn.load_weights('./models/blockage_l1/epoch_2000/variables/variables')
    l1 = l1_cnn.predict(np.array(test_data)[:,:,:,:4],verbose=1,batch_size=1)
    l1 = l1[:,:,win[0]:win[1],:]
    ref_mask = l1[:,:,:,0]<-0.5
    for i in range(3):
        if i == 0 or i == 2:
            l1[:,:,:,i][ref_mask] = -1.0
        else:
            l1[:,:,:,i][ref_mask] = 0.0
    del l1_cnn;gc.collect()
            
    #the cgan case
    cgan_cnn = unetpp((*u.SIZE['blockage'],5),base_channels=12,levels=7,growth=2)
    cgan_cnn.load_weights('./models/blockage_cgan/gen_epoch_0066000/variables/variables')
    cgan = cgan_cnn.predict(np.array(test_data),verbose=1,batch_size=1)[:,:,win[0]:win[1],:]
    ref_mask = cgan[:,:,:,0]<-0.5
    for i in range(3):
        if i == 0 or i == 2:
            cgan[:,:,:,i][ref_mask] = -1.0
        else:
            cgan[:,:,:,i][ref_mask] = 0.0
    del cgan_cnn;gc.collect()
    
    #dimensionalize and collect the results:
    def dim(x):
        fields = ['ref','vel','wid']
        for i in range(3):
            x[:,:,:,i] = u.inv_standardize(x[:,:,:,i],fields[i],'csapr')
        return x
    infilled = [dim(linear),dim(laplace),dim(telea),dim(l1),dim(cgan)]
    truth = dim(truth)
    
    #compute all of the error metrics:
    MAE, MSE, PSD, PSDt, EMD = [], [], [], [], []
    for inp in infilled:
        MSE.append(np.mean((truth-inp)**2,axis = (0,1,2)))
        MAE.append(np.mean(np.abs(truth-inp),axis = (0,1,2)))
        EMD.append(emd(inp,truth,inst='csapr'))
        PSD.append(psd(inp))
        PSDt.append(psd(inp.transpose((0,2,1,3))))
    
    true_psd = psd(truth)
    true_psdt = psd(truth.transpose((0,2,1,3)))
    
    return MAE, MSE, EMD, PSD, true_psd, PSDt, true_psdt
    
    
def compute_error_metrics(case):
    if case == 'outage':
        errors = []
        outage_sizes = [5,10,15,20,25,30,35,40]
        for sz in outage_sizes:
            errors.append(eval_outage_errors(sz))
            gc.collect()
        np.save('./data/outage_eval.npy',errors)
    elif case == 'downfill':
        downfill_sizes = [10,20,30,40,50,64]
        #i've pulled this scheme out so it can be pre-computed because it's slow:
        for ds in downfill_sizes:
            precomp_efros(ds)
        errors = []
        for ds in downfill_sizes:
            errors.append(eval_downfill_errors(ds))
            gc.collect()
        np.save('./data/downfill_eval.npy',errors)
    elif case == 'blockage':
        blockage_sizes = [4,8,12,16,20]
        errors = []
        for bsz in blockage_sizes:
            errors.append(eval_blockage_errors(bsz))
            gc.collect()
        np.save('./data/blockage_eval.npy',errors)
        
compute_error_metrics('blockage')