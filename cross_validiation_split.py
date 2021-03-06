import numpy as np

class BlockTSSplit():
    def __init__(self, n_splits, val_prop=0.2):
        self.n_splits = n_splits
        self.val_prop = val_prop

    def split(self, df):
        n_samples = len(df)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int((1 - self.val_prop) * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]
            
