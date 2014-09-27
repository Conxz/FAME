#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""A common dataset object, head motion information."""

import os
import numpy as np
from fame.base import get_runs
from fame.base import readsessid

class Headmotion:
    """Class Headmotion provides basic datatype and functions for storing and
    modifying head motion information in functional MRI.

    An instance of object Headmotion could be assigned by the instruction

        from nklib.dataset.headmotion import Headmotion

        sessdir = '/nfs/s3/workingshop/swap'
        sessid = 'S0004'
        exp = 'mt'
        rlf = 'mt.rlf'
        feat = 'func.feat'
        hm1 = Headmotion(sessdir, sessid, exp, rlf, feat)


    """

    def __init__(self, sessDir, sessID, expName, rlf, featName):
        """Initalize an instance of Headmotion.

        """
        self.sess_dir = sessDir
        self.sessID = sessID
        self.exp = expName
        self.exp_dir = os.path.join(sessDir, sessID, expName)
        self.rlf = rlf
        
        self.rlf_list = self._get_rlf()
        
        self.feat = featName
        self.mc = 'mc'
        
        self.mcf_abs_mean_f = 'prefiltered_func_data_mcf_abs_mean.rms'
        self.mcf_rel_mean_f = 'prefiltered_func_data_mcf_rel_mean.rms'
        self.mcf_abs_f = 'prefiltered_func_data_mcf_abs.rms'
        self.mcf_rel_f = 'prefiltered_func_data_mcf_rel.rms'
        self.mcf_f = 'prefiltered_func_data_mcf.par'
        
        self.mcf_rel = self._get_data(self.mcf_rel_f)
        self.mcf_abs = self._get_data(self.mcf_abs_f)
        self.mcf_rel_mean = self._get_data(self.mcf_rel_mean_f)
        self.mcf_abs_mean = self._get_data(self.mcf_abs_mean_f)
        self.mcf = self._get_data(self.mcf_f)

    
    def _get_rlf(self):
            """Get runlist.

            """
            rlf_dir = os.path.join(self.exp_dir)
            rlf_list = get_runs(self.rlf, rlf_dir)
            return rlf_list

    def _get_data(self, mcf_name):
        """Get mcf.

        Usage:
            [run_index_list, par_file_list] = sess1.getruninfo(exp_name,
                                                               run_index)

        """
        
        dat = []
        #dat_tmp = []
        for rlf in self.rlf_list:
            data_f = os.path.join(self.exp_dir, rlf, self.feat, self.mc, mcf_name)
            dat_tmp = np.loadtxt(data_f)
            
            if mcf_name == self.mcf_f:
                dat_tmp[:,0:3] = dat_tmp[:,0:3]*180.0/np.pi
            
            dat.append(dat_tmp)
        
        return dat

    def get_abs_mean(self):
        """
        """
        return self.mcf_abs_mean

    def get_abs_max(self):
        """
        """
        return np.max(self.mcf_abs, 1)

    def get_rel_mean(self):
        """
        """
        return self.mcf_rel_mean

    def get_rel_max(self):
        """
        """
        return np.max(self.mcf_rel, 1)

    def get_rel_TR_thred(self, thr):
        """Get TRs with large rel rms.

        """
        
        TRs_thred = []
        nTR = []
        
        for run in range(len(self.rlf_list)):
            TRs_thred_tmp = (self.mcf_rel[run] >= thr)
            nTR_tmp = np.sum(TRs_thred_tmp, 0)
            
            TRs_thred.append(TRs_thred_tmp)
            nTR.append(nTR_tmp)

        return nTR, TRs_thred
    
    def get_abs_TR_thred(self, thr):
        """Get TRs with large abs rms.

        """
        
        TRs_thred = []
        nTR = []

        for run in range(len(self.rlf_list)):
            TRs_thred_tmp = (self.mcf_abs[run] >= thr)
            nTR_tmp = np.sum(TRs_thred_tmp, 0)
            
            TRs_thred.append(TRs_thred_tmp)
            nTR.append(nTR_tmp)

        return nTR, TRs_thred

    def get_mcf_TR_thred(self, thr):
        """Get TRs with large head motion and the total number for each run.
        trans_x, trans_y, trans_z, rot_x, rot_y, rot_z

        """
        
        TRs_thred = []
        nTR = []
        
        for run in range(len(self.rlf_list)):
            TRs_thred_tmp = np.abs(self.mcf[run] >= thr)
            TRs_thred.append(TRs_thred_tmp)
            nTR.append(np.sum(TRs_thred_tmp, 0))

        return nTR, TRs_thred
    
    def get_mcf_max(self):
        """Get the max num for TRs in each run.

        """
        
        mcf_max = []
        
        for run in range(len(self.rlf_list)):
            mcf_max.append(np.max(np.abs(self.mcf[run]), 0))

        return mcf_max


class HeadmotionSess:
    """A class for headmotion Sess
    
    """  
    def __init__(self, sessDir, sessID_file, expName, rlfName, featName):
        """Initalize an instance of Headmotion.
        """
        self.sess_dir = sessDir
        self.sessid_list = readsessid(sessID_file)
        self.exp = expName
        self.rlf = rlfName
        self.feat = featName

        self.headmotionsess = self._get_headmotion_sess()

        #self.mcf_rel = 
        #self.mcf_abs = self._get_data(self.mcf_abs_f)
        #self.mcf_rel_mean = self._get_data(self.mcf_rel_mean_f)
        #self.mcf_abs_mean = self._get_data(self.mcf_abs_mean_f)
        #self.mcf

    def _get_headmotion_sess(self):
        """Get headmotion for sess

        """
        headmotion_sess = []

        for sess in self.sessid_list:
            hm = Headmotion(self.sess_dir, sess, self.exp, self.rlf, self.feat)
            headmotion_sess.append(hm)

        return headmotion_sess

    def get_abs_max_sess(self):
        """Get mcf abs max.

        """
        mcf_abs_max_sess = []
        for hm in self.headmotionsess:
            mcf_abs_max_sess.append(hm.get_abs_max())

        return mcf_abs_max_sess
    
    def get_abs_mean_sess(self):
        """Get mcf abs mean.

        """
        mcf_abs_mean_sess = []
        for hm in self.headmotionsess:
            mcf_abs_mean_sess.append(hm.get_abs_mean())

        return mcf_abs_mean_sess
    
    def get_rel_max_sess(self):
        """Get mcf rel max.

        """
        mcf_rel_max_sess = []
        for hm in self.headmotionsess:
            mcf_rel_max_sess.append(hm.get_rel_max())

        return mcf_rel_max_sess

    def get_rel_mean_sess(self):
        """Get mcf rel mean.

        """
        mcf_rel_mean_sess = []
        for hm in self.headmotionsess:
            mcf_rel_mean_sess.append(hm.get_rel_mean())

        return mcf_rel_mean_sess
    
    def get_rel_TR_thred_sess(self, thr):
        """Get nTRs, TRs_thred.
        rel rms

        """
        nTR_sess = []
        TR_thred_sess = []
        for hm in self.headmotionsess:
            nTR_tmp, TR_thred_tmp = hm.get_rel_TR_thred(thr)
            
            nTR_sess.append(nTR_tmp)
            TR_thred_sess.append(TR_thred_tmp)

        return nTR_sess, TR_thred_sess

    def get_abs_TR_thred_sess(self, thr):
        """Get nTRs, TRs_thred.
        abs rms

        """
        nTR_sess = []
        TR_thred_sess = []
        for hm in self.headmotionsess:
            nTR_tmp, TR_thred_tmp = hm.get_abs_TR_thred(thr)
            
            nTR_sess.append(nTR_tmp)
            TR_thred_sess.append(TR_thred_tmp)

        return nTR_sess, TR_thred_sess
    
    def get_mcf_TR_thred_sess(self, thr):
        """Get nTRs, TRs_thred.
        mcf

        """
        nTR_sess = []
        TR_thred_sess = []
        for hm in self.headmotionsess:
            nTR_tmp, TR_thred_tmp = hm.get_mcf_TR_thred(thr)
            
            nTR_sess.append(nTR_tmp)
            TR_thred_sess.append(TR_thred_tmp)

        return nTR_sess, TR_thred_sess
    
    def get_mcf_max_sess(self):
        """Get mcf max.

        """

        mcf_max_sess = []
        for hm in self.headmotionsess:
            mcf_max_sess.append(hm.get_mcf_max())

        return mcf_max_sess

