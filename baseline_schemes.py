# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:57:55 2020

@author: andrew
"""
import numpy as np
from scipy.signal import convolve2d as conv
import cv2 as cv

#subroutines used by multiple schemes:    #####################################

#finds the pixels on the edge of a mask:
def find_edge_pixels(mask):
    mask = np.pad(mask,pad_width=1,mode='reflect')
    edge = np.logical_and(mask[1:-1,1:-1],(mask[:-2,1:-1]+mask[2:,1:-1]+mask[1:-1,2:]+mask[1:-1,:-2])<4)
    return np.where(edge)

#schemes for outage and blockage cases:   #####################################
def linear(scan,mask):
    idx = np.where(mask[-1,:] == 1)[0][[0,-1]] + np.array([-1,1])
    for i in range(scan.shape[0]):
        if mask[i,idx[0]+1] == 1:
            scan[i,idx[0]:idx[1]] = np.linspace(scan[i,idx[0]],scan[i,idx[1]],int(np.diff(idx)))
    return scan

def laplace(scan, mask):
    maxiter = 1000
    #use bilinear to get an initial condition:
    scan = linear(scan, mask)
    mask[0,:], mask[-1,:], mask[:,0], mask[:,-1] = 0,0,0,0
    #uses a iterative numerical solve of laplace's eqn. to estimate the inpainting:
    TOL = 0.0001
    residual = 1.0
    stencil = np.array([[1,4,1],[4,-20,4],[1,4,1]])/50
    i = 0
    while np.max(np.abs(residual))>TOL and i < maxiter:
        i+=1
        residual = conv(scan,stencil,mode='same')*mask
        scan = scan+residual
    return scan

def telea(scan,mask):
    scan = np.uint8(scan*255)
    mask = np.uint8(mask)
    output = cv.inpaint(scan,mask,16,cv.INPAINT_TELEA)
    output = np.double(output)/255
    return output

#schemes for downfilling case:   ##############################################

def efros(scan,mask,WIN_SIZE=6):
    EH = 64 #height of the examplar sample region above blind zone
    W = WIN_SIZE*2+1
    nfill = np.sum(mask)
    
    #build a dictionary of exemplars:
    BH = np.where(mask[:,0]==0)[0][0] #top of blind zone
    sample_region = scan[BH:BH+EH,:]
    blocks = [np.zeros((W**2,))]
    pixels = [0]
    for i in range(0,EH-W,2):#WIN_SIZE//2):
        for j in range(0,scan.shape[1]-W,2):#WIN_SIZE//2):
            block = sample_region[i:i+W,j:j+W]
            if np.any(block>0.0):
                blocks.append(block.flatten())
                pixels.append(block[WIN_SIZE,WIN_SIZE])
    blocks = np.stack(blocks,axis=-1).T
    
    #fill in the masked region
    while np.any(mask):
        IE, JE = find_edge_pixels(mask)
        pad_scan = np.pad(scan,WIN_SIZE,'reflect')
        pad_mask = 1.0-np.pad(mask,WIN_SIZE,'constant',constant_values=0)
        perc = np.int(100*np.sum(mask)/nfill)
        for ie, je in zip(IE,JE):
            scan_region = pad_scan[ie:ie+W,je:je+W].flatten()
            mask_region = pad_mask[ie:ie+W,je:je+W].flatten()
            scores = np.sum(((blocks-scan_region)**2)*mask_region,axis=1)/np.sum(mask_region)
            best = np.argmin(scores)
            scan[ie,je] = pixels[best]
            mask[ie,je] = 0.0
    return scan

def efros3(scan,mask,WIN_SIZE=4):
    scan = np.copy(scan)
    mask = np.copy(mask)
    EH = 64 #height of the examplar sample region above blind zone
    W = WIN_SIZE*2+1
    nfill = np.sum(mask)
    
    #build a dictionary of exemplars:
    BH = np.where(mask[:,0]==0)[0][0] #top of blind zone
    sample_region = scan[BH:BH+EH,:,:]
    blocks = [np.concatenate((np.ones((W,W,1))*-1,np.zeros((W,W,1)),np.ones((W,W,1))*-1),axis=2).flatten()]
    pixels = [np.array([-1.0,0.0,-1.0])]
    for i in range(0,EH-W,2):#WIN_SIZE//2):
        for j in range(0,scan.shape[1]-W,2):#WIN_SIZE//2):
            block = sample_region[i:i+W,j:j+W,:]
            if np.any(block[:,:,0]>-1.0):
                blocks.append(block.flatten())
                pixels.append(block[WIN_SIZE,WIN_SIZE,:])
    blocks = np.stack(blocks,axis=-1).T
    
    #fill in the masked region
    while np.any(mask):
        IE, JE = find_edge_pixels(mask)
        pad_scan = np.pad(scan,[[WIN_SIZE,WIN_SIZE],[WIN_SIZE,WIN_SIZE],[0,0]],'reflect')
        pad_mask = 1.0-np.pad(mask,WIN_SIZE,'constant',constant_values=0)
        for ie, je in zip(IE,JE):
            scan_region = pad_scan[ie:ie+W,je:je+W,:].flatten()
            mask_region = pad_mask[ie:ie+W,je:je+W]
            mask_region = np.stack((mask_region,mask_region,mask_region),axis=-1).flatten()
            scores = np.sum(((blocks-scan_region)**2)*mask_region,axis=1)/np.sum(mask_region)
            best = np.argmin(scores)
            scan[ie,je,:] = pixels[best]
            mask[ie,je] = 0.0
    return scan

def marching_avg(scan,mask,WIN_SIZE=8):
    while np.any(mask):            #while there are still pixels to inpaint
        #get a list of the pixels on the border of the inpainting:
        IE, JE = find_edge_pixels(mask)
        #create a padded version of the scan and mask
        pad_scan = np.pad(scan,WIN_SIZE,'reflect')
        pad_mask = 1.0-np.pad(mask,WIN_SIZE,'reflect')#flip the zeros and ones, so there are ones over pixels with valid data
        W = WIN_SIZE*2+1
        for ie, je in zip(IE,JE):
            scan_region = pad_scan[ie:ie+W,je:je+W]
            mask_region = pad_mask[ie:ie+W,je:je+W]
            masked_conv = np.sum(scan_region*mask_region)/np.sum(mask_region)
            scan[ie,je] = masked_conv
            mask[ie,je] = 0.0
    return scan

def repeat(scan,mask):
    BH = np.where(mask[:,0]==0)[0][0]
    sample = scan[BH,:]
    for i in range(BH-1,-1,-1):
        scan[i,:] = sample
    return scan