import os
import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Callable, Optional

class TensorDataSet(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        x_column: str,
        y_column: Optional[str] = None,
        tfms: Callable = nn.Identity(),
        root: Optional[str] = None,
        preload: bool = False,
        load_key: Optional[str] = None,
    ):
        self.tfms = tfms
        self.preload = preload
        self.load_key = load_key
        
        if root is not None:
            rootify = lambda p: os.path.join(root, p)
        else:
            rootify = lambda p: p
            
        exists = frame[x_column].apply(os.path.exists)
        if exists.sum() != len(frame):
            print("Frame has non-existant slides:")
            print(frame[x_column][~exists])
            frame = frame.copy()[exists]
        
        if preload:
            if self.load_key is not None:
                loader = lambda p: torch.load(rootify(p))[self.load_key]
            else:
                loader = lambda p: torch.load(rootify(p))
        else:
            loader = lambda p: rootify(p)
        
        self.x = frame[x_column].apply(loader).values
        self.y = None
        
        if y_column is not None:
            self.y = np.asarray(frame[y_column].values)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        
        if not self.preload:
            x = torch.load(x)
            if self.load_key is not None:
                x = x[self.load_key]
        
        y = -1000
        if self.y is not None:
            y = self.y[idx]
            y = y[None, ...]
    
        return self.tfms(x), y