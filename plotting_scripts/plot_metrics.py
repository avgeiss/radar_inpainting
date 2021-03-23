# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 17:43:20 2020

@author: andrew
"""

import numpy as np
import matplotlib.pyplot as plt

fields = ['Reflectivity','Doppler Velocity','Spectral Width']

#plotting function for scalar metrics:
def plot_error(x,y,labels,units,legend):
    plt.figure(figsize=(15,5),dpi=300)
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.plot(x,y[:,:,i],'-o',linewidth=2)
        plt.grid(color=[0.9,0.9,0.9])
        dx = (x[-1]-x[0])*0.03
        plt.xlim([x[0]-dx,x[-1]+dx])
        plt.title(fields[i])
        plt.xlabel(labels[0])
        plt.ylabel(labels[1] + ' ' + units[i])
    
    plt.legend(legend)
        
def plot_psd(hy,hyt,fh,vy,vyt,fv,leg,labels):
    plt.figure(figsize=(10,4.6),dpi=300)
    plt.subplot(1,2,1)
    plt.plot(fv,vy,linewidth=2)
    plt.plot(fv,vyt[:,0],'k-',linewidth=2)
    plt.xlim([fv[0],fv[-1]])
    plt.ylabel(labels[0][2])
    plt.title(labels[0][0])
    plt.xlabel(labels[0][1])
    plt.grid(color=[0.9,0.9,0.9])
    
    
    plt.subplot(1,2,2)
    plt.plot(fh,hy,linewidth=2)
    plt.plot(fh,hyt[:,0],'k-',linewidth=2)
    plt.xlim([fh[0],fh[-1]])
    plt.legend(leg)
    plt.ylabel(labels[1][2])
    plt.title(labels[1][0])
    plt.xlabel(labels[1][1])
    plt.grid(color=[0.9,0.9,0.9])
    

      
#do the kazr outage case:
metrics = np.load('../data/outage_eval.npy',allow_pickle=True)

#plot the error metrics:
leg = ['Linear', 'Laplace', 'Telea', 'L1 CNN', 'CGAN']
x = np.array([5,10,15,20,25,30,35,40])*2*2/60
labels = ['Outage Duration $(min)$','Mean Abs. Error']
mae = np.array([sz[0] for sz in metrics])
units = ['$(dBZ)$','$(ms^{-1})$','$(ms^{-1})$']
plot_error(x,mae,labels,units,leg)
plt.savefig('../figures/eval/outage_mae.png')
plt.close('all')

mse = np.array([sz[1] for sz in metrics])
labels = ['Outage Duration $(min)$','Mean Squared Error']
units = ['$(dBZ)^2$','$(ms^{-1})^2$','$(ms^{-1})^2$']
plot_error(x,mse,labels,units,leg)
plt.savefig('../figures/eval/outage_mse.png')
plt.close('all')

emd = np.array([sz[2] for sz in metrics])
labels = ['Outage Duration $(min)$','Earth Mover\'''s Distance']
units = ['$(dimensionless)$','','']
plot_error(x,emd,labels,units,leg)
plt.savefig('../figures/eval/outage_emd.png')
plt.close('all')

#psd plot
fh = np.linspace(1,32,31)/(64*2/60) #osc per minute
fv = np.linspace(1,128,127)/(256*30/1000) #osc per km
leg = ['Linear', 'Laplace', 'Telea', 'L1 CNN', 'CGAN', 'Truth']
labels = [['Reflectivity Vertical PSD','Frequency $(km^{-1})$','Power Spectral Density $(dBZ)$'],
          ['Reflectivity Horizontal PSD','Frequency $(min^{-1})$','Power Spectral Density $(dBZ)$']]
plot_psd(np.array(metrics[-1][5])[:,:,0].T,metrics[-1][6],fh,
          np.array(metrics[-1][3])[:,:,0].T,metrics[-1][4],fv,
          leg,labels)
plt.savefig('../figures/eval/outage_ref_psd.png')





#do the kazr downfilling case:
metrics = np.load('./data/downfill_eval.npy',allow_pickle=True)

#plot the error metrics:
leg = ['Marching AVG.', 'Repeat', 'Efros', 'L1 CNN', 'CGAN']
x = np.array([10,20,30,40,50,64])*30/1000
labels = ['Blind Zone Height $(km)$','Mean Abs. Error']
mae = np.array([sz[0] for sz in metrics])
units = ['$(dBZ)$','$(ms^{-1})$','$(ms^{-1})$']
plot_error(x,mae,labels,units,leg)
plt.savefig('./figures/eval/downfill_mae.png')
plt.close('all')

mse = np.array([sz[1] for sz in metrics])
labels = ['Blind Zone Height $(km)$','Mean Squared Error']
units = ['$(dBZ)^2$','$(ms^{-1})^2$','$(ms^{-1})^2$']
plot_error(x,mse,labels,units,leg)
plt.savefig('./figures/eval/downfill_mse.png')
plt.close('all')

emd = np.array([sz[2] for sz in metrics])
labels = ['Blind Zone Height $(km)$','Earth Mover\'''s Distance']
units = ['$(dimensionless)$','','']
plot_error(x,emd,labels,units,leg)
plt.savefig('./figures/eval/downfill_emd.png')
plt.close('all')

# psd plot
fv = np.linspace(1,32,31)/(64*30/1000) #osc per km
fh = np.linspace(1,128,127)/(2*256/60) #osc per min
leg = ['Marching AVG.', 'Repeat', 'Efros', 'L1 CNN', 'CGAN','Truth']
labels = [['Reflectivity Vertical PSD','Frequency $(km^{-1})$','Power Spectral Density $(dBZ)$'],
          ['Reflectivity Horizontal PSD','Frequency $(min^{-1})$','Power Spectral Density $(dBZ)$']]
plot_psd(np.array(metrics[-1][5])[:,:,0].T,metrics[-1][6],fh,
          np.array(metrics[-1][3])[:,:,0].T,metrics[-1][4],fv,
          leg,labels)
plt.savefig('./figures/eval/downfill_ref_psd.png')



#do the csapr infilling case:
metrics = np.load('./data/blockage_eval.npy',allow_pickle=True)

#plot the error metrics:
leg = ['Linear', 'Laplace', 'Telea', 'L1 CNN', 'CGAN']
x = np.array([4,8,12,16,20])*2
labels = ['Blockage Size $(deg)$','Mean Abs. Error']
mae = np.array([sz[0] for sz in metrics])
units = ['$(dBZ)$','$(ms^{-1})$','$(ms^{-1})$']
plot_error(x,mae,labels,units,leg)
plt.savefig('./figures/eval/blockage_mae.png')
plt.close('all')

mse = np.array([sz[1] for sz in metrics])
labels = ['Blockage Size $(deg)$','Mean Squared Error']
units = ['$(dBZ)^2$','$(ms^{-1})^2$','$(ms^{-1})^2$']
plot_error(x,mse,labels,units,leg)
plt.savefig('./figures/eval/blockage_mse.png')
plt.close('all')

emd = np.array([sz[2] for sz in metrics])
labels = ['Blockage Size $(deg)$','Earth Mover\'''s Distance']
units = ['$(dimensionless)$','','']
plot_error(x,emd,labels,units,leg)
plt.savefig('./figures/eval/blockage_emd.png')
plt.close('all')

#psd plot
fh = np.linspace(1,16,15)/32 #osc per minute
fv = np.linspace(1,512,511)/(1024*50/1000) #osc per km
leg = ['Linear', 'Laplace', 'Telea', 'L1 CNN', 'CGAN', 'Truth']
labels = [['Reflectivity Radial PSD','Frequency $(km^{-1})$','Power Spectral Density $(dBZ)$'],
          ['Reflectivity Azimuthal  PSD','Frequency $(deg^{-1})$','Power Spectral Density $(dBZ)$']]
plot_psd(np.array(metrics[-1][5])[:,:,0].T,metrics[-1][6],fh,
          np.array(metrics[-1][3])[:,:,0].T,metrics[-1][4],fv,
          leg,labels)
plt.savefig('./figures/eval/blockage_ref_psd.png')


