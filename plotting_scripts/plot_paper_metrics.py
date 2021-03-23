#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:48:28 2021

@author: andrew
"""
import numpy as np
import matplotlib.pyplot as plt


#plotting function for scalar metrics:
def plot_error(x,y,title,labels,units):
    plt.plot(x,y,'-o',linewidth=2)
    plt.grid(color=[0.9,0.9,0.9])
    dx = (x[-1]-x[0])*0.03
    plt.xlim([x[0]-dx,x[-1]+dx])
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1] + ' ' + units[i])
        
def plot_psd(hy,hyt,fh,vy,vyt,fv,leg,labels):
    plt.subplot(2,3,5)
    plt.plot(fv,vy,linewidth=2)
    plt.plot(fv,vyt,'k-',linewidth=2)
    plt.xlim([fv[0],fv[-1]])
    plt.ylabel(labels[0][2])
    plt.title(labels[0][0])
    plt.xlabel(labels[0][1])
    plt.grid(color=[0.9,0.9,0.9])
    
    
    plt.subplot(2,3,6)
    plt.plot(fh,hy,linewidth=2)
    plt.plot(fh,hyt,'k-',linewidth=2)
    plt.xlim([fh[0],fh[-1]])
    plt.legend(leg)
    plt.ylabel(labels[1][2])
    plt.title(labels[1][0])
    plt.xlabel(labels[1][1])
    plt.grid(color=[0.9,0.9,0.9])
    

#outage case##################################################################
metrics = np.load('../data/outage_eval.npy',allow_pickle=True)
plt.figure(figsize=(18,12))
leg = ['Linear', 'Laplace', 'Telea', 'L1 CNN', 'CGAN']
x = np.array([5,10,15,20,25,30,35,40])*2*2/60
labels = ['Outage Duration $(min)$','Mean Abs. Error']
mae = np.array([sz[0] for sz in metrics])
units = ['$(dBZ)$','$(ms^{-1})$','$(ms^{-1})$']
titles = ['a) Reflectivity MAE','b) Doppler Velocity MAE','c) Spectrum Width MAE']
for i in range(3):
    plt.subplot(2,3,i+1)
    plot_error(x,mae[:,:,i],titles[i],labels,units)
emd = np.array([sz[2] for sz in metrics])
labels = ['Outage Duration $(min)$','Earth Mover\'''s Distance']
units = ['$(dimensionless)$','','']
plt.subplot(2,3,4)
plot_error(x,emd[:,:,0],'d) Reflectivity EMD',labels,units)
plt.legend(leg)
fh = np.linspace(1,32,31)/(64*2/60) #osc per minute
fv = np.linspace(1,128,127)/(256*30/1000) #osc per km
leg = ['Linear', 'Laplace', 'Telea', 'L1 CNN', 'CGAN', 'Truth']
labels = [['e) Reflectivity Vertical PSD','Frequency $(km^{-1})$','Power Spectral Density $(dBZ)$'],
          ['f) Reflectivity Horizontal PSD','Frequency $(min^{-1})$','Power Spectral Density $(dBZ)$']]
plot_psd(np.array(metrics[-1][5])[:,:,0].T,metrics[-1][6],fh,
          np.array(metrics[-1][3])[:,:,0].T,metrics[-1][4],fv,
          leg,labels)
plt.savefig('../figures/paper/fig_04.png',dpi=600,bbox_inches='tight',pad_inches=0.1)

#creates the extra PSD plots, need to change the subplot commands in psd plotter:
# plt.close()
# fids = ['ref','vel','wid']
# for i in range(3):
#     plt.figure(figsize=(12,6))
#     plot_psd(np.array(metrics[-1][5])[:,:,i].T,metrics[-1][6][:,i],fh,
#               np.array(metrics[-1][3])[:,:,i].T,metrics[-1][4][:,i],fv,
#               leg,labels)
#     plt.savefig('../figures/eval/outage_' + fids[i] + '_psd.png')


#downfill case################################################################
metrics = np.load('../data/downfill_eval.npy',allow_pickle=True)
plt.figure(figsize=(18,12))
leg = ['Marching AVG.', 'Repeat', 'Efros', 'L1 CNN', 'CGAN']
x = np.array([10,20,30,40,50,64])*30/1000
labels = ['Blind Zone Height $(km)$','Mean Abs. Error']
mae = np.array([sz[0] for sz in metrics])
units = ['$(dBZ)$','$(ms^{-1})$','$(ms^{-1})$']
titles = ['a) Reflectivity MAE','b) Doppler Velocity MAE','c) Spectrum Width MAE']
for i in range(3):
    plt.subplot(2,3,i+1)
    plot_error(x,mae[:,:,i],titles[i],labels,units)
emd = np.array([sz[2] for sz in metrics])
labels = ['Blind Zone Height $(km)$','Earth Mover\'''s Distance']
units = ['$(dimensionless)$','','']
plt.subplot(2,3,4)
plot_error(x,emd[:,:,0],'d) Reflectivity EMD',labels,units)
plt.legend(leg)
fv = np.linspace(1,32,31)/(64*30/1000) #osc per km
fh = np.linspace(1,128,127)/(2*256/60) #osc per min
leg = ['Marching AVG.', 'Repeat', 'Efros', 'L1 CNN', 'CGAN','Truth']
labels = [['e) Reflectivity Vertical PSD','Frequency $(km^{-1})$','Power Spectral Density $(dBZ)$'],
          ['f) Reflectivity Horizontal PSD','Frequency $(min^{-1})$','Power Spectral Density $(dBZ)$']]
plot_psd(np.array(metrics[-1][5])[:,:,0].T,metrics[-1][6],fh,
          np.array(metrics[-1][3])[:,:,0].T,metrics[-1][4],fv,
          leg,labels)
plt.savefig('../figures/paper/fig_06.png',dpi=600,bbox_inches='tight',pad_inches=0.1)


# #creates the extra PSD plots, need to change the subplot commands in psd plotter:
# plt.close()
# fids = ['ref','vel','wid']
# for i in range(3):
#     plt.figure(figsize=(12,6))
#     plot_psd(np.array(metrics[-1][5])[:,:,i].T,metrics[-1][6][:,i],fh,
#               np.array(metrics[-1][3])[:,:,i].T,metrics[-1][4][:,i],fv,
#               leg,labels)
#     plt.savefig('../figures/eval/downfill_' + fids[i] + '_psd.png')


#csapr case####################################################################
metrics = np.load('../data/blockage_eval.npy',allow_pickle=True)
plt.figure(figsize=(18,12))
leg = ['Linear', 'Laplace', 'Telea', 'L1 CNN', 'CGAN']
x = np.array([4,8,12,16,20])*2
labels = ['Blockage Size $(deg)$','Mean Abs. Error']
mae = np.array([sz[0] for sz in metrics])
units = ['$(dBZ)$','$(ms^{-1})$','$(ms^{-1})$']
titles = ['a) Reflectivity MAE','b) Doppler Velocity MAE','c) Spectrum Width MAE']
for i in range(3):
    plt.subplot(2,3,i+1)
    plot_error(x,mae[:,:,i],titles[i],labels,units)
plt.legend(leg)
emd = np.array([sz[2] for sz in metrics])
labels = ['Blockage Size $(deg)$','Earth Mover\'''s Distance']
units = ['$(dimensionless)$','','']
plt.subplot(2,3,4)
plot_error(x,emd[:,:,0],'d) Reflectivity EMD',labels,units)
plt.legend(leg)
fh = np.linspace(1,16,15)/32 #osc per minute
fv = np.linspace(1,512,511)/(1024*50/1000) #osc per km
leg = ['Linear', 'Laplace', 'Telea', 'L1 CNN', 'CGAN', 'Truth']
labels = [['e) Reflectivity Radial PSD','Frequency $(km^{-1})$','Power Spectral Density $(dBZ)$'],
          ['f) Reflectivity Azimuthal  PSD','Frequency $(deg^{-1})$','Power Spectral Density $(dBZ)$']]
plot_psd(np.array(metrics[-1][5])[:,:,0].T,metrics[-1][6][:,0],fh,
          np.array(metrics[-1][3])[:,:,0].T,metrics[-1][4][:,0],fv,
          leg,labels)
plt.savefig('../figures/paper/fig_08.png',dpi=600,bbox_inches='tight',pad_inches=0.1)


#creates the extra PSD plots, need to change the subplot commands in psd plotter:
# plt.close()
# fids = ['ref','vel','wid']
# for i in range(3):
#     plt.figure(figsize=(12,6))
#     plot_psd(np.array(metrics[-1][5])[:,:,i].T,metrics[-1][6][:,i],fh,
#               np.array(metrics[-1][3])[:,:,i].T,metrics[-1][4][:,i],fv,
#               leg,labels)
#     plt.savefig('../figures/eval/outage_' + fids[i] + '_psd.png')