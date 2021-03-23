import numpy as np
from scipy.signal import convolve2d
from skimage.measure import label

def unfold(vel,nyq,periodic = False):
    #for now, deals with large negative velocities that have been aliased to 
    #positive values. need to modify in the future to deal with general case,
    #but this handles the vast majority (99%+) of velocity aliasing.
    #
    # Inputs:
    # vel - 2D matrix of doppler velocity data
    # nyq - The nyquist velocity (maximum representable speed)
    #
    # Returns:
    # corrected_vel - 2D matrix the size of 'vel'
    # mask - a mask of corrected regions
    
    def sort_labels(regions):
        nr = int(np.max(regions)+1)
        sizes = []
        for i in range(1,nr):
            sizes.append(np.sum(regions==i))
        mapping = np.flip(np.argsort(sizes))
        rsorted = np.zeros(regions.shape)
        for i in range(1,nr):
            rsorted[regions==mapping[i-1]+1] = i
        return rsorted
    
    def segment(vel):
        #segment positive and negative velocity regions separately
        regions_pos, nr_pos = label(vel>0,connectivity=2,background=0,return_num=True)
        regions_neg, nr_neg = label(vel<0,connectivity=2,background=0,return_num=True)
        regions_neg = regions_neg+nr_pos
        regions_neg[regions_neg == nr_pos] = 0
        regions = regions_pos + regions_neg
        
        #combines edge regions assuming axis 1 is periodic
        if periodic:
            lr,lv = regions[:,1], vel[:,1]
            rr,rv = regions[:,-2], vel[:,-2]
            for i in range(lr.size):
                if lr[i]!=0 and rr[i]!=0 and lr[i]!=rr[i] and abs(lv[i]-rv[i])<nyq:
                    old_lbl = np.copy(rr[i])
                    regions[regions == rr[i]] = lr[i]
                    regions[regions>old_lbl] -= 1
                    rr = regions[:,-2]
                    
        return sort_labels(regions)
    
    def get_exterior(mask):
        filt = np.array([[0,1,0],[1,0,1],[0,1,0]])
        ext = np.logical_and(np.logical_not(mask),
                             convolve2d(np.float16(mask),filt,mode='same',fillvalue=0)>0)
        return ext
    
    def neighbor_info(regions,vel,rlabel):
        #returns the ids of each unique region neighboring region 'rlabel', along 
        #with the number of pixels along the border and the magnitude of the change
        #in velocity along the border
        #
        #neighbor_labels, border_length, delta_v = neighbor_info(regions,vel,rlabel)
        
        #firstly trim to the region of interest to make things faster:
        if not periodic or not (np.any(regions[:,1]==rlabel) and np.any(regions[:,-2]==rlabel)):
            bbx = np.where(np.any(regions==rlabel,axis=1))[0][[0,-1]] + [-1,2]
            bby = np.where(np.any(regions==rlabel,axis=0))[0][[0,-1]] + [-1,2]
            regions = regions[bbx[0]:bbx[1],bby[0]:bby[1]]
            vel = vel[bbx[0]:bbx[1],bby[0]:bby[1]]
        
        #find the pixels on the exterior of the region:
        ext = get_exterior(regions==rlabel)
        
        #make a list of the neighboring region labels:
        neighbors = list(np.unique(regions[ext]))
        if 0 in neighbors:
            neighbors.remove(0)
        
        #get some stats about each of the neighboring regions:
        labels, lengths, deltas = [],[],[]
        for nbr_lbl in neighbors:
            nbr_ext = get_exterior(regions==nbr_lbl)
            nbr_mn = np.mean(vel[regions*ext==nbr_lbl])
            r_mn = np.mean(vel[regions*nbr_ext==rlabel])
            labels.append(nbr_lbl)
            lengths.append(np.sum(regions*ext==nbr_lbl))
            deltas.append(nbr_mn-r_mn)
            
        return np.array(labels), lengths, np.array(deltas)
    
    def merge_fold(regions,vel):
        rlabels = list(np.unique(regions))
        while len(rlabels)>1:
            # if len(rlabels)<4:
            #     print('Almost done')
            rlabel = rlabels.pop()
            nlabel, nblen, dvbar = neighbor_info(regions,vel,rlabel)
            if nlabel.size>0:
                idx = np.argmax(nblen)
                nlabel, dvbar = nlabel[idx], dvbar[idx]
                if np.abs(dvbar)>nyq:
                    vel[regions==rlabel] += 2*nyq*np.sign(dvbar)*np.floor((np.abs(dvbar)+nyq)/(2*nyq))
                regions[regions==rlabel] = nlabel
                regions[regions>=rlabel] -= 1
        return vel, regions
    
    #zero pad the velocities:
    vel = np.pad(vel,[[1,1],[1,1]],mode='constant',constant_values=0)
    
    #segment all the contiguous velocity regions:
    regions = segment(vel)
    
    #merge neighboring regions that have folding:
    vel, regions = merge_fold(regions, vel)
    
    #center each remaining region so |vbar|<nyq (as long as its bigger than a few pixels)
    for r in range(1,int(np.max(regions))):
        if np.sum(regions==r)>3:
            vmd = np.median(vel[regions==r])
            vel[regions==r] -= 2*nyq*np.sign(vmd)*np.floor((np.abs(vmd)+nyq)/(2*nyq))
    
    return vel[1:-1,1:-1]