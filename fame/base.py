# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Base functions and classes for nkpi 

"""

import os
import sys
import csv
import glob
import logging
import time

import nibabel as nib
from nibabel import load
from nibabel import save
import numpy as np
import scipy as sp
import random
from scipy.signal import detrend


import numpy as np
import scipy.io as sio
import os

def npy2mat(npy_in, mat_out):
    """Save npy/npz to mat.
    
    Contribution:
    -------------
    Author: kongxiangzheng@gmail.com
    Date: 04/08/2013
    Editors: [plz add own name after edit here]
    
    """
    
    dat = np.load(npy_in)
    sio.savemat(mat_out, dat)
    if os.path.exists(mat_out):
        return True

def get_runs(rlf, rlfdir):
    """Get the run set
    
    Parameters
    ----------
    rlf : string
        run list file
    rlfdir : string
        parent dir
    
    Contributions
    -------------
    Date : 
    Author : yangzetian
    Reviewer :
    
    """
    
    if rlf: 
        rlf_file = os.path.join(rlfdir, rlf)
    else:
        rlf_file = ''
    if rlf_file and os.path.isfile(rlf_file):
        with open(rlf_file, 'r') as f:
            runs = f.read().split()
    else:
        runs = [os.path.basename(x)
                for x in glob.glob(os.path.join(rlfdir, '0*'))
                if os.path.isdir(x)]
    return runs


def get_logger(logfile):
    """Get a unified logger for all pynit bin

    Parameters
    ----------
    logfile : logger file name

    Notes
    -----
    Adapted from `Logging Cookbook` by yangzetian
    
    """
    logger = logging.getLogger('pynit')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logfile, 'w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.ERROR)
    formatter_fh = logging.Formatter(fmt='%(asctime)s %(levelname)s: '+
                   '%(message)s', datefmt='%m-%d %H:%M')
    formatter_ch = logging.Formatter('%(levelname)s: %(message)s')
    fh.setFormatter(formatter_fh)
    ch.setFormatter(formatter_ch)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class nkpi_logger(logging.getLoggerClass()):
    """Get a unified logger for all nitk-pipeline modula.
    
    To ensure that installing a customised Logger class will not undo
    customisations already applied by other code.
    http://docs.python.org/dev/library/logging.html#logging.getLoggerClass

    Parameters
    ----------
    logfile : logger file name

    Notes
    -----
    Adapted from `Logging Cookbook` by yangzetian&Xiangzhen Kong
    
    """
    def __init__(self, logfile):
        self.name = 'nkpi'
        self.handlers = []
        self.setLevel(logging.DEBUG)
        self.disabled=0
        self.filters = []
        self.propagate = 1
        self.parent = None
        
        if logfile == None:
            logfile = os.path.join(get_dotnkpi(),'nkpi_' + get_timestr() + '.log')
        
        fh = logging.FileHandler(logfile, 'w')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.ERROR)
        
        formatter_fh = logging.Formatter(fmt='%(asctime)s %(levelname)s: '+
                       '%(message)s', datefmt='%m-%d %H:%M')
        fh.setFormatter(formatter_fh)
        formatter_ch = logging.Formatter('%(levelname)s: %(message)s')
        ch.setFormatter(formatter_ch)
        
        self.addHandler(fh)
        self.addHandler(ch)

def get_timestr():
    """Get the time str for now.
    
    Returns
    -------
        timestr, for example, 20130308133703.

    """
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

def get_dotnkpi():
    """Get the ~/.nkpi dir.

    Returns
    -------
        dotnkpi, that is, '~/.nkpi'

    """
    userhome = os.path.expanduser('~')
    dotnkpi = os.path.join(userhome, '.nkpi')
    if not os.path.exists(dotnkpi):
        os.mkdir(dotnkpi)

    return dotnkpi


def subjrlf(subject_id):
    """Get the run set
    
    Parameters
    ----------
    subject_id : string
        subject_id
    
    Contributions
    -------------
    Date : 
    Author : 
    Reviewer :
    
    """
    
    fsesspar = '/nfs/s2/nsptesting/resting/examples/sesspar'
    fsesspar = open(fsesspar)
    data_dir = fsesspar.readline().strip()
    rlf = open(os.path.join(data_dir,subject_id,'rest','rfMRI.rlf'))
    run_list = [line.strip() for line in rlf]
    info = dict(func=[[subject_id,'rest',run_list,'rest']],
                brain=[[subject_id,'3danat','reg','brain']],
                struct=[[subject_id,'3danat','reg','T1']])
    return info

def detrend4d(para):
    """docs
    
    Parameters
    ----------
    para : 
        
    
    Contributions
    -------------
    Date : 
    Author : 
    Reviewer :

    """

    intnorm_file = para[0][0]
    mask_file = para[0][1]
    imgseries = load(intnorm_file)
    mask = load(mask_file)
    voxel_timecourses = imgseries.get_data()
    m = np.nonzero(mask.get_data())
    for a in range(len(m[1])):
        voxel_timecourses[m[0][a],m[1][a],m[2][a]] = detrend(voxel_timecourses[m[0][a],m[1][a],m[2][a]], type='linear') 
    imgseries.data = voxel_timecourses
    outfile = para[0][0] + '_dt.nii.gz'
    imgseries.to_filename(outfile)
    return outfile

def getrank(data):
    """docs
    
    Parameters
    ----------
    para : 
        
    
    Contributions
    -------------
    Date : 2012.7.15
    Author : WangXu
    Reviewer :

    """
    temp = data.argsort(axis=-1)
    rank = temp.argsort(axis=-1)
    return rank

def kendall(ts):
    """docs
    
    Parameters
    ----------
    para : 
        
    
    Contributions
    -------------
    Date : 2012.7.15
    Author : WangXu
    Reviewer :

    """
    import numpy as np
    [k,n] = np.shape(ts)
    RS = np.sum(ts,0)
    S = np.sum(RS*RS)-np.sum(RS)*np.sum(RS)/n
    F = k*k*(n*n*n-n)
    W = 12*float(S)/float(F)
    return W

def readsess(fsesspar, fsessid):

    """read session parent dir file and session indetifier file

    input:
        fsesspar: file for session parent dir
        fsessid:  file for session indentifier
    output:
        sessid: a list for session indentifier in full path

    """

    fsesspar= open(fsesspar)
    sesspar = fsesspar.readline().strip()
    fsessid = open(fsessid)
    sessid  = [line.strip() for line in fsessid]
    return sesspar, sessid


def readsessid(fsessid):

    """Read session indetifier file.

    input:
        fsessid:  file for session indentifier
    output:
        sessid: a list for session indentifier

    """

    fsessid = open(fsessid)
    sessid  = [line.strip() for line in fsessid]
    return sessid


def readsesspar(fsesspar, fsessid):

    """Read session parent dir file.

    input:
        fsesspar: file for session parent dir
    output:
        sesspar: a list for session parent dir

    """

    fsesspar= open(fsesspar)
    sesspar = fsesspar.readline().strip()
    return sesspar

def randomsplit(list, n):
    """Split the input list into n equal parts.
    
    Inputs:
        list:
        n:
    Outputs:
        
    
    """
    
    random.shuffle(list)
    randparts = np.array_split(array(list), n)
    return randparts

def prepdataset(roi_mean):

    """Get correlation vector for one subject.

    input:
        roi_mean:  the mean time series for the rois
    output:
        subjcorr:  the 1D corr vector
     Author: Wang Xu
     Date: 2012.03
    """
    corrmtx = np.corrcoef(roi_mean)
    corrvect = []
    for i in range(np.shape(corrmtx)[0]-1):
        j = i + 1
        corrvect = corrvect + corrmtx[i][j:].tolist()
        subjcorr = ' '.join(['%s'% el for el in corrvect])
    return subjcorr,corrmtx

def featurename(roinamefile):
    
    """Get feature names from the roinamefile.

    input:
        roinamefile:  file for contain the rois' names
    output:
        fnames: a list for feature names
    Author: Wang Xu
    Date: 2012.03
    """
    roinamefile = open(roinamefile)	
    roinames  = [line.strip() for line in roinamefile]
    fnames = []
    for i in roinames:
        index = 1
        for j in roinames[index:]:
            fname = roinames[i] + '_' + roinames[j]
            fnames.append(fname)
        index = index + 1
    return fnames

class sfcsv():
    
    """A csvfile class.
    
    Parameters
    ----------
    filename : str
        A csvfile path.
    attrname: list
        A list of attributes in the csvfile.
    prefix: str
        A prefix of these outputs, default 'mvpa'.
    ndim: int
        The number of volumes for each sample, default 5.
    colsess: str
        The headname of sessid column in the scvfile, default 'SID'.
    
    Contributions
    -------------
    Date: 2012-03-23
    Author: Xiangzhen Kong
    Reviewer :
    
    Notes
    -----
    
   
    """
    
    def __init__(self, filename, attrnames, prefix = 'mvpa', ndim = 1, colsess = 'SID'):
        
        self.name = filename
        self.attrnames = attrnames
        self.prefix = prefix
        self.ndim = ndim
        self.sesslist = []
        
        fcsv = open(self.name, 'rb')
        reader = csv.reader(fcsv)
        for line in reader:
            if reader.line_num == 1:
                colnames = line
                break
        reader = csv.DictReader(open(self.name), colnames)
        
        self.attrval = [[] for i in range(len(self.attrnames))]
        for line in reader:
            if reader.line_num == 1:
               continue
            
            for i in range(len(self.attrnames)):
                attrval = line.get(self.attrnames[i])
                self.attrval[i].append(attrval)
            
            sessid = line.get(colsess)
            self.sesslist.append(sessid)
        
    def write_sessid(self):
        """Write out sessid file.
        
        """
        sessfile = self.prefix + '_sessid'
        if os.path.exists(sessfile):
            print 'Warning: ',sessfile,'exists!'
            exit(-1)
        fsess = open(sessfile, 'a')
        for sess in self.sesslist:
            fsess.write(sess + '\n')
        fsess.close()
    
    def write_attribute(self):
        """Write out a attribute file.
        
        """
        i = 0
        for attr in self.attrnames:
            attrfile = self.prefix + 'attr_' + attr
            fattr = open(attrfile, 'a')
        
            j = 0
            for val in self.attrval[i]:
                for k in range(self.ndim):
                    fattr.write(val + ' ' + str(j) + '\n')
                j += 1
            fattr.close()
            i += 1



class Niireader:
    """Module readimage provides to load the image to array.

    Parameters:
    -----------
        path: data file
        maskpath: an atlas file
        
    Example:
    --------

    >>> import niiread

    Contributions
    -------------
    Date: 2012-03
    Author: Yangyue Huang
    Reviewer: Xiangzhen Kong
    
    Notes
    -----

    """
    
    def __init__(self, path, maskpath=None, threshold=None):
        """Initial.
        
        """
        
        self.img = nib.load(path)
        self.data = self.img.get_data()
        
        if len(self.data.shape) == 3:
            self.n = 1
            self.p = self.data.size
            self.fldata = self.data.flatten()
        elif len(self.data.shape) == 4:
            self.n = self.data.shape[3]
            self.p = self.data[:, :, :, 0].size
            self.fldata = np.array([self.data[:, :, :, i].flatten()
                                        for i in range(self.n)])
        
        if not maskpath is None:
            self.imgmask = nib.load(maskpath)
            self.mask = self.imgmask.get_data()
            self.flmask = self.mask.flatten()

        if not threshold is None:
            self.flmask[np.mean(self.fldata,0)<threshold]=0
        
        self.label = np.unique(self.flmask)
        self.label = self.label[self.label != 0]
        self.label.sort()

    def getroi_mean(self):
        """Return mean for ROIs."""
        
        x1 = self.fldata
        x2 = self.flmask
        n = self.n
        if len(self.data.shape) == 3:
            roi_mean = 0      
            for i in self.label:
                roi_mean = np.r_[roi_mean, np.mean(x1[x2==i])]
            roi_mean = roi_mean[1:]
        elif len(self.data.shape) == 4:
            roi_mean = np.zeros(self.n)      
            for i in self.label:
                roi_mean = np.c_[roi_mean, np.mean(x1[:, x2==i], 1)]
            roi_mean = roi_mean[:, 1:]      
        return roi_mean

    def getroi_max(self):
        """Return max in ROIs."""
        
        x1 = self.fldata
        x2 = self.flmask
        if len(self.data.shape) == 3:
            roi_max = 0      
            for i in self.label:
                roi_max = np.r_[roi_max, np.max(x1[x2==i])]
            roi_max = roi_max[1:]
        elif len(self.data.shape) == 4:
            roi_max = np.zeros(self.n)      
            for i in self.label:
                roi_max = np.c_[roi_max, np.max(x1[:, x2==i], 1)]
            roi_max = roi_max[:, 1:]       
        return roi_max
    

class Niireader2:
    """Module readimage provides to load the image to array.

    Parameters:
    -----------
        path: data file
        maskpath: an atlas file
        
    Example:
    --------

    >>> import niiread

    Contributions
    -------------
    Date: 2012-03
    Author: Yangyue Huang
    Reviewer: Xiangzhen Kong
    
    Notes
    -----

    """
    
    def __init__(self, path):
        """Initial.
        
        """
        
        self.img = nib.load(path)
        self.data = self.img.get_data()
        
        if len(self.data.shape) == 3:
            self.n = 1
            self.p = self.data.size
            self.fldata = self.data.flatten()
        elif len(self.data.shape) == 4:
            self.n = self.data.shape[3]
            self.p = self.data[:, :, :, 0].size
            self.fldata = np.array([self.data[:, :, :, i].flatten()
                                        for i in range(self.n)])
        
    def get_mask_label(self, maskfile, threshold=None):
        if not maskfile is None:
            self.imgmask = nib.load(maskfile)
            self.mask = self.imgmask.get_data()
            self.flmask = self.mask.flatten()

        if not threshold is None:
            self.flmask[np.mean(self.fldata,0)<threshold]=0
        
        self.label = np.unique(self.flmask)
        self.label = self.label[self.label != 0]
        self.label.sort()

    def getroi_mean(self, maskfile):
        """Return mean for ROIs."""
        
        self.get_mask_label(maskfile)
        x1 = self.fldata
        x2 = self.flmask
        n = self.n
        if len(self.data.shape) == 3:
            roi_mean = [0]      
            for i in self.label:
                roi_mean = np.r_[roi_mean, np.mean(x1[x2==i])]
            if len(roi_mean)>1:
                roi_mean = roi_mean[1:]
            else:
                roi_mean = roi_mean[0:]
        elif len(self.data.shape) == 4:
            roi_mean = np.zeros(self.n)      
            for i in self.label:
                roi_mean = np.c_[roi_mean, np.mean(x1[:, x2==i], 1)]
            roi_mean = roi_mean[:, 1:]      
        return roi_mean

    def getroi_max(self, maskfile):
        """Return max in ROIs."""
        
        self.get_mask_label(maskfile)
        x1 = self.fldata
        x2 = self.flmask
        if len(self.data.shape) == 3:
            roi_max = 0      
            for i in self.label:
                roi_max = np.r_[roi_max, np.max(x1[x2==i])]
            roi_max = roi_max[1:]
        elif len(self.data.shape) == 4:
            roi_max = np.zeros(self.n)      
            for i in self.label:
                roi_max = np.c_[roi_max, np.max(x1[:, x2==i], 1)]
            roi_max = roi_max[:, 1:]       
        return roi_max

def get_roisize(atlasfile):
    """
    """
    if not atlasfile is None:
        atlasimg = nib.load(atlasfile)
        atlasdat = atlasimg.get_data()
        atlasdat_fl = atlasdat.flatten()
        
        labels = np.unique(atlasdat_fl)
        labels = labels[labels != 0]
        labels.sort()
        
        if len(labels != 0):
            sizelist = np.zeros(len(labels))
            for i in range(len(labels)):
                sizelist[i] =  atlasdat_fl[atlasdat_fl==labels[i]].shape[0]

        else:
            sizelist = [0]
         
        return sizelist
    else:
        print atlasfile, ' NOT exists!'
