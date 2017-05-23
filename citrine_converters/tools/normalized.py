from __future__ import division

import numpy as np


class Normalized(np.ndarray):
    def __new__(cls, vec):
        obj = np.copy(vec).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        if hasattr(obj, 'lower') and hasattr(obj, 'range'):
            self.lower = obj.lower
            self.range = obj.range
        else:
            # normalize
            lower = obj.min()
            upper = obj.max()
            self.lower = lower
            self.range = upper - lower
            # self = (self - lower)/(upper - lower)
            self -= lower
            self /= self.range

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    @property
    def unscaled(self):
        """Returns the unscaled data (initial range)"""
        return self*self.range + self.lower
#end 'class Normalizer(object):'
