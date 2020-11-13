"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import collections
import pandas as pd


def tensor2seq(input_sequence):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_sequence, np.ndarray):
        if isinstance(input_sequence, torch.Tensor):  # get the data from a variable
            sequence_tensor = input_sequence.data
        else:
            return input_sequence
        sequence_numpy = sequence_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        #if sequence_numpy.shape[0] == 1:  # grayscale to RGB
        #    sequence_numpy = np.tile(sequence_numpy, (3, 1, 1))
        #sequence_numpy = (np.transpose(sequence_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        sequence_numpy = input_sequence
    return decode_oneHot(sequence_numpy)


def save_sequence(tensorSeq):
    i = 0
    results =collections.OrderedDict()
    results['realA'] = []
    results['fakeB'] = []
    results['realB'] = []
    for seqT in tensorSeq:
        for label, seq in seqT.items():
            seq = tensor2seq(seq)
            results[label].append(seq)
            #im.save(dir + label + '_' + str(batch) + '_' + str(i) + '.png', quality=100)
            i = i + 1
    results = pd.DataFrame(results)
    results.to_csv('results.csv', index=False)


def decode_oneHot(seq):
    keys = ['A', 'T', 'C', 'G', 'M']
    dSeq = ''
    for i in range(np.size(seq, 1)):
        pos = np.argmax(seq[:, i])
        dSeq += keys[pos]
    return dSeq