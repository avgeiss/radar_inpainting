import numpy as np
from netCDF4 import Dataset
from glob import glob
from vel_dealias import unfold
from multiprocess import Pool
from matplotlib import colors
import matplotlib.pyplot as plt

BUF_RANGE = [1,17]
min_weather_size = 10
DOWNFILL_CUT_RANGE=[7,67]
OUTAGE_RANGE = [4,42]
BLOCKAGE_WIDTH = [4,21]
BLOCKAGE_START = [16,256]
TEST_FRAC = 0.2
NYQ = {'kazr':8.0, 'csapr':16.5}
INST = {'outage':'kazr','downfill':'kazr','blockage':'csapr'}
SIZE = {'outage': [256,256], 'downfill': [256,256], 'blockage': [1024,128]}
DATA_RANGE = {'kazr': {'ref': [-10,40], 'vel': [-NYQ['kazr']*1.5,NYQ['kazr']*1.5], 'wid':[0,2.5]},
              'csapr': {'ref': [0,60], 'vel': [-NYQ['csapr']*2,NYQ['csapr']*2], 'wid':[0,5.5]}}
data_dir = './data/'
raw_data_dir = '/home/andrew/Data/'
inds = np.load(data_dir + 'downfill_sample_inds.npy')
DOWNFILL_SAMPLE_INDS = inds[:int(len(inds)*(1-TEST_FRAC))]
inds = np.load(data_dir + 'outage_sample_inds.npy')
OUTAGE_SAMPLE_INDS = inds[:int(len(inds)*(1-TEST_FRAC))]

#  STANDARDIZATION   ########################################################
def standardize(x,field,instrument):
    mn,mx = DATA_RANGE[instrument][field]
    if field == 'ref':
        x[np.abs(x)>100] = -50.0
        x = 1.5*(x-mn)/(mx-mn)-0.5
        x[x<=-0.5] = -1.0
        x[x>1.0] = 1.0
    elif field == 'vel':
        x[np.abs(x)>100] = 0.0
        x = np.tanh(1.5*x/mx)
    elif field == 'wid':
        x[np.abs(x)>100] = 0.0
        x = 2.0*x/mx-1.0
        x[x>1.0] = 1.0
        x[x<-1.0] = -1.0
    return x

def inv_standardize(x,field,instrument):
    mn,mx = DATA_RANGE[instrument][field]
    if field == 'ref':
        x[x<-0.5] = -0.5
        x = (x+0.5)*(mx-mn)/1.5+mn
    elif field == 'vel':
        x = np.arctanh(x)*mx/1.5
    elif field == 'wid':
        x = mx*(x+1.0)/2.0
    return x
    
def boxcar1d(x,N):
    filtered = np.zeros((x.shape[0]-N+1,x.shape[1]),x.dtype)
    mn = np.sum(x[:N,:],axis=0)
    for i in range(x.shape[0]-N):
        filtered[i,:] = mn/N
        mn += x[i+N,:]-x[i,:]
    filtered[i+1,:] = mn/N
    return filtered

def boxcar2d(x,N):
    x = boxcar1d(x,N)
    x = boxcar1d(x.T,N).T
    return x

#   DATA PREPROCESSING   #####################################################
def ingest_kazr():
    #converts kazr data from the NetCDF files retrieved from ARM to npy files used
    #by neural networks. Performs standardization and velocity delaiasing
    
    #get a list of the files:
    nc_dir = raw_data_dir + 'CACTI_KAZR'
    out_dir = data_dir
    files = glob(nc_dir + '/*.nc')
    files.sort()
    
    def proc_nc_file(f):
        print('Processing data from: ' + f)
        dset = Dataset(f)
        
        x = dset['reflectivity'][:].data.T
        
        #if there's nothing in the file, dont use it:
        weather = x>DATA_RANGE['kazr']['ref'][0]
        w_frac = np.mean(np.double(weather))
        l_count = np.sum(weather[:,:2])
        r_count = np.sum(weather[:,-2:])
        if w_frac<0.01 and l_count<10 and r_count<10:
            return None
        
        x = dset['reflectivity'][:].data.T
        x = np.float16(x)
        ref = standardize(x,'ref','kazr')
        mask = ref<-0.8
        
        x = dset['mean_doppler_velocity'][:].data.T
        x = np.double(x)
        x[mask] = 0.0
        x[np.abs(x)>NYQ['kazr']] = 0.0
        x = unfold(x,NYQ['kazr'])
        x = standardize(x,'vel','kazr')
        vel = np.float16(x)
        
        x = dset['spectral_width'][:].data.T
        x = np.float16(x)
        x = standardize(x,'wid','kazr')
        x[mask] = -1.0
        wid = x
        
        return np.stack((ref,vel,wid),axis=-1)[:SIZE['outage'][0],:,:]
    
    p = Pool(24)
    data = p.map(proc_nc_file,files,chunksize=1)
    p.close()
    data2 = []
    for d in data:
        if not d is None:
            data2.append(d)
    data = np.concatenate(data2,axis=1)
        
    np.save(out_dir + '/kazr.npy',data)

def get_kazr_sample_inds():
    #gets a list of the starting indices that are appropriate for downfilling
    #and inpainting
    print('Loading Data...')
    mask = np.load(data_dir + 'kazr.npy')[:,:,0]
    mask = np.float16(mask>-1.0)
    
    print('Computing Convolution...')
    mask = boxcar2d(mask,min_weather_size)>0.99
    
    #the indices for the outage period case:
    #enforce that the sample has weather
    print('Computing Outage Mask...')
    N = SIZE['outage'][1]
    outage_mask = np.any(mask,axis=0)
    outage_mask = boxcar1d(np.float16(outage_mask[:-N,np.newaxis]),N)
    outage_inds = np.where(outage_mask[:,0]>0)[0]
    np.save(data_dir + 'outage_sample_inds.npy',outage_inds)
    
    #the indices for the downfilling cases:
    #enforce that the bottom 1/4 of the sample has weather
    print('Computing Downfilling Mask...')
    downfill_mask = np.any(mask[DOWNFILL_CUT_RANGE[0]:DOWNFILL_CUT_RANGE[1],:],axis=0)
    downfill_mask = boxcar1d(np.float16(downfill_mask[:,np.newaxis]),SIZE['downfill'][1])
    downfill_inds = np.where(downfill_mask[:,0]>0)[0]
    np.save(data_dir + 'downfill_sample_inds.npy',downfill_inds)
    
def ingest_csapr():
    print('Loading Data...')
    csapr_dir = raw_data_dir + 'CACTI_CSAPR/'
    data = np.stack((np.load(csapr_dir + 'taranis_attenuation_corrected_reflectivity.npy'),
                     np.load(csapr_dir + 'mean_doppler_velocity.npy'),
                     np.load(csapr_dir + 'spectral_width.npy'),
                     np.load(csapr_dir + 'composite_mask.npy')),axis=-1)
    data = data.transpose(0,2,1,3)
    
    #only save the 1500 scans with the most weather:
    mask_frac = np.mean(data[:,:,:,-1]>0.0,axis=(1,2))
    inds = np.flip(np.argsort(mask_frac))[:1500]
    inds = np.sort(inds)
    data = data[inds,:,:,:]
    
    def proc_sweep(data):
        index = data[0]
        data = data[1]
        print('Processing sweep: ' + str(index))
        ref = data[:,:,0]
        ref = standardize(ref,'ref','csapr')
        mask = data[:,:,-1]
        mask = np.logical_or(ref<-0.8,mask==0)
        ref[mask] = -1.0
        wid = data[:,:,2]
        wid = standardize(wid,'wid','csapr')
        wid[mask] = -1.0
        vel = data[:,:,1]
        vel[mask] = 0.0
        vel[np.abs(vel)>NYQ['csapr']] = 0.0
        vel = unfold(vel,NYQ['csapr'],periodic=True)
        vel = standardize(vel,'vel','csapr')
        return np.stack((ref,vel,wid),axis=-1)
    
    print('Prepping data')
    data = list(data)
    index = list(np.arange(len(data)))
    print('Starting pool')
    p = Pool(24)
    data = p.map(proc_sweep,zip(index,data),chunksize=1)
    p.close()
    data = np.array(data)
    
    np.save(data_dir + 'csapr.npy',data)
    
#gets the test dataset
def make_kazr_test_sets():
    for case in ['outage','downfill']:
        data = np.load(data_dir + INST[case] + '.npy')
        inds = np.load(data_dir + case + '_sample_inds.npy')
        inds = inds[int(-len(inds)*TEST_FRAC):]
        test_set = []
        N = SIZE[case][0]
        last = -N
        for ind in inds:
            if ind-last > N//2:
                test_set.append(data[:,ind:ind+N,:])
                last = ind
        np.save('./data/' + case + '_test_set.npy',test_set)
        
def make_csapr_test_set():
    case = 'blockage'
    samps_per_sweep = 3
    data = np.load(data_dir + 'csapr.npy')
    N = int(data.shape[0]*TEST_FRAC)
    data = data[-N:,:,:,:]
    samples = np.zeros((N*samps_per_sweep,*SIZE[case],3),dtype='float16')
    for i in range(N):
        for j in range(samps_per_sweep):
            sweep = np.copy(data[i,...])
                
            #find angles with weather, and center a random one:
            weather_frac = np.mean(sweep[:,:,0] > -1.0,axis=0)+0.001
            pick_prob = weather_frac/np.sum(weather_frac)
            blockage_ind = np.random.choice(np.arange(0,sweep.shape[1]),p=pick_prob)
            sweep = np.roll(sweep,sweep.shape[1]//2-blockage_ind,axis=1)
            
            #trim to the correct size:
            NA = SIZE[case][1]
            MID = sweep.shape[1]//2
            sweep = sweep[:SIZE[case][0],MID-NA//2:MID+NA//2,:]
            
            #randomly flip wrt azimuth:
            if np.random.choice([True,False]):
                sweep = np.flip(sweep,axis=1)
                
            samples[i*samps_per_sweep + j,...] = sweep
    np.save('./data/' + case + '_test_set.npy',samples)
            
    
#   MINIBATCH GENERATORS   ###################################################

def downfill_batch(x,data = None,sample_inds=DOWNFILL_SAMPLE_INDS,cut_range=DOWNFILL_CUT_RANGE,buf_range=BUF_RANGE):
    
    #get size information about requested data:
    BS = x.shape[0]
    NT = x.shape[2]
    
    if not data is None:
        for i in range(BS):
            idx = np.random.choice(sample_inds)
            x[i,:,:,:3]  = np.copy(data[:,idx:idx+NT,:])
            if np.random.choice([True, False]):
                x[i,:,:,:] = np.flip(x[i,:,:,:],axis=1)
        
    FS = min_weather_size
    for i in range(BS):
        #find the valid altitudes where a cut can be made:
        if len(cut_range)==2:
            mask = np.float16(x[i,cut_range[0]:cut_range[1]+FS,:,0]>-0.5)
            mask = boxcar2d(mask,FS)>0.99
            valid_levs = np.where(np.any(mask,axis=1))[0]
            if len(valid_levs)>0:
                cut_ind = np.random.choice(valid_levs)+cut_range[0]
            else:
                cut_ind = np.random.randint(cut_range[0],cut_range[1])
        else:
            cut_ind = cut_range[0]
        #make a mask to indicate where to paint:
        N_buf = np.random.randint(buf_range[0],buf_range[1])
        buf = np.linspace(1.0,0.0,N_buf+2)[1:-1]
        mask = np.zeros((x.shape[1],x.shape[2]),dtype='float16')
        mask[:cut_ind,:] = 1.0
        mask[cut_ind:cut_ind+N_buf,:] = buf[:,np.newaxis]
        x[i,:,:,-1] = mask
        
def outage_batch(x,data = None,sample_inds=OUTAGE_SAMPLE_INDS,win_range=OUTAGE_RANGE,buf_range=BUF_RANGE):
    BS = x.shape[0]
    NT = x.shape[2]
    for i in range(BS):
        if not data is None:
            #get a random sample of KAZR obs:
            start_idx = np.random.choice(sample_inds)
            x[i,:,:,:3] = np.copy(data[:,start_idx:start_idx+NT,:])
            if np.random.choice([True, False]):
                x[i,:,:,:] = np.flip(x[i,:,:,:],axis=1)
            
        #select the size of the window to block out:
        win_size = np.random.randint(win_range[0],win_range[1]+1)
        win = [NT//2-win_size,NT//2+win_size]
        #create a mask:
        N_buf = np.random.randint(buf_range[0],buf_range[1]+1)
        buf = np.linspace(1.0,0.0,N_buf+2)[1:-1]
        mask = np.zeros((x.shape[1],x.shape[2]))
        mask[:,win[0]:win[1]] = 1.0
        mask[:,win[0]-N_buf:win[0]] = np.flip(buf[np.newaxis,:],axis=1)
        mask[:,win[1]:win[1]+N_buf] = buf[np.newaxis,:]
        x[i,:,:,-1] = mask

def blockage_batch(x,data=None):
    BS = x.shape[0]         #batch size
    
    for i in range(BS):
        if not data is None:
            #select a random sweep:
            NS = data.shape[0]      #number of sweeps
            sweep = np.copy(data[np.random.randint(0,NS),:SIZE['blockage'][0],...])
            
            #find angles with weather, and center a random one:
            weather_frac = np.mean(sweep[:,:,0] > -1.0,axis=0)+0.001
            pick_prob = weather_frac/np.sum(weather_frac)
            blockage_ind = np.random.choice(np.arange(0,sweep.shape[1]),p=pick_prob)
            sweep = np.roll(sweep,sweep.shape[1]//2-blockage_ind,axis=1)
            
            #trim to the correct size:
            NA = SIZE['blockage'][1]
            MID = sweep.shape[1]//2
            sweep = sweep[:SIZE['blockage'][0],MID-NA//2:MID+NA//2,:]
            
            #randomly flip wrt azimuth:
            if np.random.choice([True,False]):
                sweep = np.flip(sweep,axis=1)
            
            #targets are done
            x[i,:,:,:3] = sweep

        #select the size and start range of the blockage:
        start = np.random.randint(BLOCKAGE_START[0],BLOCKAGE_START[1]+1)
        width = np.random.randint(BLOCKAGE_WIDTH[0],BLOCKAGE_WIDTH[1]+1)
        
        #add in the blockage
        MID = SIZE['blockage'][1]//2
        
        #make the mask
        mask = np.zeros(SIZE['blockage'],dtype='float16')
        mask[start:,MID-width:MID+width] = 1.0
    
        #fuzzify the mask:
        N_buf = np.random.randint(BUF_RANGE[0],BUF_RANGE[1])
        buf = np.linspace(1.0,0.0,N_buf+2)[1:-1]
        corner = np.flip(np.outer(buf,buf.T),axis=0)
        mask[start:,MID+width:MID+width+N_buf] = buf[np.newaxis,:]
        mask[start:,MID-width-N_buf:MID-width] = np.flip(buf[np.newaxis,:],axis=1)
        mask[start-N_buf:start,MID-width:MID+width] = np.flip(buf[:,np.newaxis],axis=0)
        mask[start-N_buf:start,MID+width:MID+width+N_buf] = corner
        mask[start-N_buf:start,MID-width-N_buf:MID-width] = np.flip(corner,axis=1)
        x[i,:,:,3] = mask

def gen_batch(batch_fun,x,data=None):
    #create a regular batch with a random seed as the last input channel:
    batch_fun(x[:,:,:,:4],data)
    sz = x.shape
    x[:,:,:,4] = np.random.normal(0.0,scale=0.5,size=sz[:-1])
    
    #also output some soft labels to use a targets:
    return np.random.uniform(0.0,0.1,size=(sz[0],1))

def dis_batch(batch_fun,x,data,gen):
    
    #the first half of the batch will be generator outputs:
    N = x.shape[0]//2
    gbatch = np.zeros((N,x.shape[1],x.shape[2],x.shape[3]+1))
    gen_batch(batch_fun,gbatch,data)
    gout = gen.predict(gbatch)
    x[:N,:,:,:3] = gout              #the generator outputs
    x[:N,:,:,3] = gbatch[:,:,:,3]    #the mask
    
    #the second half is real data:
    real = np.zeros((N,x.shape[1],x.shape[2],x.shape[3]))
    batch_fun(real,data)
    x[N:,:,:,:] = real
    
    #return soft labels:
    return np.concatenate((np.random.uniform(0.9,1.0,size=(N,1)),
                           np.random.uniform(0.0,0.1,size=(N,1))),axis=0)
    

BATCH_FUNC = {'outage': outage_batch,'downfill': downfill_batch,'blockage': blockage_batch}


########### DEFINE PLOTTING FUNCTIONS  ########################################


#define some constants:
FWID=7
fields = ['ref','vel','wid']
csapr_ranges = DATA_RANGE['csapr']
kazr_ranges = DATA_RANGE['kazr']
titles = {'ref': 'Reflectivity (dBZ)', 'vel': 'Doppler Velocity (m/s)', 'wid': 'Spectrum Width (m/s)'}
csapr_res = [1,0.1]
kazr_res = [2/60,0.03]

#create a special colormap for reflectivity:
cmap = plt.cm.get_cmap('gist_ncar',256)
cmap = cmap(np.linspace(0,1,256))
cmap[:1, :] = np.array([0.8,0.8,0.8,1])
ref_cmap = colors.ListedColormap(cmap)

#create a special colormap for spectral width:
cmap = plt.cm.get_cmap('RdPu',256)
cmap = cmap(np.linspace(0,1,256))
cmap[:1, :] = np.array([0.8,0.8,0.8,1])
wid_cmap = colors.ListedColormap(cmap)

colormaps = {'ref': ref_cmap, 'vel': 'seismic', 'wid': wid_cmap}

#plotting for individual CSAPR fields:
def plot_csapr_field(data,field_name):
    data = np.double(data)
    data = inv_standardize(data,field_name,'csapr')
    
    #get data type specific variables:
    drng = csapr_ranges[field_name]
    dmn, dmx = drng[0],drng[1]
    colormap = colormaps[field_name]
    
    #make the figure
    rng = np.arange(0,data.shape[0],dtype='double')*csapr_res[1]
    azm = np.arange(0,data.shape[1],dtype='double')*csapr_res[0]
    if data.shape[1] == 360:
        azm = np.linspace(0,360,360)
    [azm,rng] = np.meshgrid(azm,rng)
    x = np.cos(azm*(np.pi/180))*rng
    y = np.sin(azm*(np.pi/180))*rng
    im = plt.pcolor(x,y,data,cmap=colormap,vmin=dmn,
                    vmax=dmx)
    plt.ylabel('(km)')
    plt.xlabel('(km)')
    mx = max(np.max(x),np.max(y))
    mn = min(np.min(x),np.min(y))
    plt.ylim([mn,mx])
    plt.xlim([mn,mx])
        
    #add colorbar and labels:
    plt.colorbar(im)
    plt.title(titles[field_name])
    
def plot_csapr(data,field_name='ref',fname=None):
    data = np.copy(data)
    if data.ndim == 2:
        f = plt.figure(figsize=[FWID,FWID*0.8])
        plot_csapr_field(data,field_name)
    elif data.ndim == 3:
        f = plt.figure(figsize=[FWID*3,FWID*0.75])
        for i in range(3):
            plt.subplot(1,3,i+1)
            plot_csapr_field(data[:,:,i],fields[i])
    if fname is not None:
        plt.savefig(fname,dpi=500)
        plt.close(f)

#plotting function:
def plot_kazr_field(data,field_name='ref'):
    data = np.double(data)
    data = inv_standardize(data,field_name,'kazr')
    
    dmn, dmx = kazr_ranges[field_name][0], kazr_ranges[field_name][1]
    colormap = colormaps[field_name]
        
    #define the axes:
    extent = (0,data.shape[1]*kazr_res[0],0,data.shape[0]*kazr_res[1])
    aspect = kazr_res[0]/kazr_res[1]
    
    #make the figure:
    im = plt.imshow(data,origin='lower',cmap=colormap,vmin=dmn,
                vmax=dmx, extent=extent, aspect=aspect)
    
    #add colorbar and labels:
    plt.colorbar(im)
    plt.ylabel('Altitude (km)')
    plt.xlabel('Time (min)')
    plt.title(titles[field_name])

def plot_kazr(data,field_name='ref',fname=None):
    data = np.copy(data)
    if data.ndim == 2:
        f = plt.figure(figsize=[FWID,FWID*0.8])
        plot_kazr_field(data,field_name)
    elif data.ndim == 3:
        f = plt.figure(figsize=[FWID*3,FWID*0.75])
        for i in range(3):
            plt.subplot(1,3,i+1)
            plot_kazr_field(data[:,:,i],fields[i])
    if fname is not None:
        plt.savefig(fname,dpi=500)
        plt.close(f)
        
def show():
    plt.show()

def plot(data,case,field_name='ref',fname=None):
    #helper function, given an inpainting scenario ('case') will call the appropriate plotting function
    if case == 'outage' or case == 'downfill':
        plot_kazr(data,field_name,fname)
    elif case == 'blockage':
        plot_csapr(data,field_name,fname)

def create_sample_sets(case):
    #steps through data from each of the test sets and allows the user to
    #select 10 examples to be output during training to qualitative evaluation
    N = 5#number of samples to get
    size = SIZE[case]
    data = np.load(data_dir + INST[case] + '.npy')
    batch = BATCH_FUNC[case]
    samples_x = []
    x = np.zeros((2,*size,4))
    while len(samples_x)<N:
        batch(x,data)
        plot(x[0,:,:,:3],case)
        show()
        plt.imshow(np.double(x[0,:,:,3]))
        show()
        response = input('Add to sample set? (y)')
        if response == 'y':
            samples_x.append(np.copy(x[0,...]))
    np.save('./data/' + case + '_samples.npy',np.array(samples_x))