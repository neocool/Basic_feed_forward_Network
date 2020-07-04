import random
from statistics import mean
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import math
from Classes.network_class import network


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = network(4,1,500).to(device)
print(model)
