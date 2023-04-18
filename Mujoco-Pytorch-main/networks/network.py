import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base import Network

from emlp.groups import SO,O,S,Z
from emlp.reps import T,V
import emlp.nn.pytorch as enn
import jax ,jaxlib

from utils.utils import state2state , StateActionMapping, Map

class Actor(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.tanh,last_activation = None, trainable_std = False):
        super(Actor, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        self.trainable_std = trainable_std
        if self.trainable_std == True:
            self.logstd = nn.Parameter(torch.zeros(1, output_dim))
    def forward(self, x):
        mu = self._forward(x)
        if self.trainable_std == True:
            std = torch.exp(self.logstd) 
        else:
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)
        return mu,std

class Critic(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation = None):
        super(Critic, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        
    def forward(self, *x):
        x = torch.cat(x,-1)
        return self._forward(x)
    


class EActor(nn.Module):
    def __init__(self,device,trainable_std = False):
        super(EActor,self).__init__()
        
        self.device = device
        self.map = Map(self.device)
        self.trainable_std = trainable_std
        self.repin = 17*V
        self.repout = 4*V**0
        self.G = S(n=4)
        self.layer1 = enn.EMLP(rep_in=self.repin,rep_out=self.repout,group=self.G).to(self.device)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(4,128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,8)
        ).to(self.device)

        self.trainable_std = trainable_std
        if self.trainable_std == True:
            self.logstd = nn.Parameter(torch.zeros(1,8)).to(self.device)

    def forward(self,x):
        x = self.map.BatchStateMap(x)
        # print(f"Shape after mapping:{x.shape}")
        x = self.layer1(x)
        # print(f"Shape after the first layer:{x.shape}")
        mu = self.layer2(x)
        if self.trainable_std == True:
            std = torch.exp(self.logstd) 
        else:
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)
        return mu,std
    
class ECritic(nn.Module):
    def __init__(self,device):
        super(ECritic, self).__init__()
        self.device = device
        self.map = Map(self.device)
        self.repin = 19*V
        self.repout = 4*V**0
        self.G = S(n=4)
        self.layer1 = enn.EMLP(rep_in=self.repin,rep_out=self.repout,group=self.G).to(self.device)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(4,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1)
        ).to(self.device)
        
    def forward(self,state,action):
        x = self.map.BatchStateActionMap(state,action)
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        return x 
            
