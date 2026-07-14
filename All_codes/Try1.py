import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

data = pd.read_excel("/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Diabetic_wound_lactate.xlsx")
print(data.head())

