"""¶àÄ£Ì¬ÈÚºÏ""" 
import numpy as np 
class HybridFusion: 
    def fuse(self, eeg, fmri): 
        return np.concatenate([eeg, fmri]) 
