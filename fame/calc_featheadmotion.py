#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import argparse
import numpy as np
#import scipy.io as sio

from fame.base import npy2mat
from fame.featheadmotion import HeadmotionSess
from fame.base import nkpi_logger


def main():
    parser = argparse.ArgumentParser(
            description = 'A cmd-line for getting head motion.')
    
    parser.add_argument('-sd',
                        dest = 'sessdir',
                        required = True,
                        metavar = 'sess-dir',
                        help = 'The sessdir.')
    parser.add_argument('-sf',
                        dest = 'sessfile',
                        required = True,
                        metavar = 'sess-file',
                        help = 'The sess list file.')
    parser.add_argument('-exp',
                        dest = 'exp',
                        required = True,
                        metavar = 'exp-name',
                        help = 'The exp name.')
    parser.add_argument('-rlf',
                        dest = 'rlf',
                        required = True,
                        metavar = 'runlist-file',
                        help = 'The run list file.')
    parser.add_argument('-feat',
                        dest = 'feat',
                        required = True,
                        metavar = 'feat-dir',
                        help = 'The feat dir.')
    parser.add_argument('-thr',
                        dest = 'thr',
                        type = float,
                        nargs = 2,
                        required = True,
                        metavar = 'thr',
                        help = 'The thresh for motion, absthr relthr.')
    parser.add_argument('-o',
                        dest = 'output',
                        required = True,
                        metavar = 'output',
                        help = 'The output file.')

    parser.add_argument('--log', 
                        dest = 'log', 
                        default = None,
                        metavar = 'log-file',
                        help='log name for the processing.')
    parser.add_argument('-v', action='version', version='%(prog)s 0.0.1')

    args = parser.parse_args()

    logger = nkpi_logger(args.log)
    logger.debug(args)
    logger.info('Start Running')
    
    hms = HeadmotionSess(args.sessdir, args.sessfile, args.exp, args.rlf, 
            args.feat)
    
    sessid_list = hms.sessid_list
    abs_max_sess = hms.get_abs_max_sess()
    abs_mean_sess = hms.get_abs_mean_sess()
    abs_nTR_sess, abs_TR_thred_sess = hms.get_abs_TR_thred_sess(args.thr[0])
    
    rel_max_sess = hms.get_abs_max_sess()
    rel_mean_sess = hms.get_abs_mean_sess()
    rel_nTR_sess, rel_TR_thred_sess = hms.get_rel_TR_thred_sess(args.thr[1])

    
    np.savez_compressed(args.output + '.npz', 
                                datsrc = args.sessdir,
                                sessid = sessid_list,
                                
                                abs_max=abs_max_sess, 
                                abs_mean=abs_mean_sess,
                                
                                rel_max=rel_max_sess, 
                                rel_mean=rel_mean_sess,

                                abs_thr=args.thr[0],
                                abs_nTR=abs_nTR_sess,

                                rel_thr=args.thr[1],
                                rel_nTR=rel_nTR_sess
                                )
    
    
    npy2mat(args.output + '.npz', args.output + '.mat')
    #dat = np.load(args.output + '.npz')
    #sio.savemat(args.output + '.mat', dat)

    logger.info('Finished Running')
    
if __name__ == '__main__':
    main()

