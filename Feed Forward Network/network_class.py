import torch
import torch.nn as nn

class network(nn.Module):
	def __init__(self,D_IN,D_OUT,hidden_nodes):
		super(network, self).__init__()
						
		#Network 1 
		self.input1 = nn.Linear(D_IN,hidden_nodes,bias=True).to(device)
		self.hidden1 = nn.Linear(hidden_nodes,hidden_nodes,bias=True).to(device)
		self.hidden2 = nn.Linear(hidden_nodes,hidden_nodes,bias=True).to(device)
		self.hidden3 = nn.Linear(hidden_nodes,hidden_nodes,bias=True).to(device)
		self.hidden4 = nn.Linear(hidden_nodes,hidden_nodes,bias=True).to(device)
		self.output1 = nn.Linear(hidden_nodes,D_OUT,bias=True).to(device)
		
	def forward(self,x):
		nn1_pred = self.input1(x)
		nn1_pred = self.hidden1(nn1_pred)
		nn1_pred = self.hidden2(nn1_pred)
		nn1_pred = self.hidden3(nn1_pred)
		nn1_pred = self.hidden4(nn1_pred)
		nn1_pred = self.output1(nn1_pred)
		return nn1_pred
