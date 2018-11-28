'''
Utils file for transformer network parameters
Built by Yash Bonde for team msaic_yash_sara_kuma_6223

Cheers!
'''

import numpy as np

def load_numpy_array(filepath):
	'''
	return loaded numpy array
	'''
	return np.load(filepath)

def add_padding(inp, pad_id, seqlen):
    '''
    Pad the input values to the required length
    Args:
        inp: input sequence with shape (batch_size, <variable>)
        pad_id: (int) ID for padding element
    Returns:
    	(batch_size, seqlen)
    	NOTE: zeros are added at the starting
    '''
    sequences = []
    for s in sequences:
        s = np.append(arr = s, values = np.ones(seqlen - len(s)))
        sequences.append(s)
    return sequences