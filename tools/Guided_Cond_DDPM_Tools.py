# This script provides a set of tools for creating a guided and/or conditional tabular DDPM Model:

import numpy as np
import json
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

import sklearn.preprocessing as PP

'''
==========================================
Set up the data normalizer class
==========================================

'''        

class Data_Normalizer:
    def __init__(self, X_LL_Scaled, X_UL_Scaled,datalength):
        
        self.normalizer = PP.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(datalength // 30, 1000), 10),
            subsample=int(1e9)
            )
        
        self.X_LL_Scaled = X_LL_Scaled
        self.X_UL_Scaled = X_UL_Scaled
        
        self.X_LL_norm = np.zeros((1,len(X_LL_Scaled)))
        self.X_UL_norm = np.zeros((1,len(X_LL_Scaled)))
        
        self.X_mean = np.zeros((1,len(X_LL_Scaled)))
        self.X_std = np.zeros((1,len(X_LL_Scaled)))
        
    def fit_Data(self,X):
        
        
        
        x = 2.0*(X-self.X_LL_Scaled)/(self.X_UL_Scaled- self.X_LL_Scaled) - 1.0
        
        self.normalizer.fit(x)
        x = self.normalizer.transform(x) # Scale Dataset between 
        #x = (X-self.X_LL_Scaled)/(self.X_UL_Scaled- self.X_LL_Scaled)
        

        return x
    
    def transform_Data(self,X):
        x = 2.0*(X-self.X_LL_Scaled)/(self.X_UL_Scaled- self.X_LL_Scaled) - 1.0
        
        
        x = self.normalizer.transform(x)
        return x
        

    def scale_X(self,z):
        #rescales data
        z = self.normalizer.inverse_transform(z)
        scaled = (z + 1.0) * 0.5 * (self.X_UL_Scaled - self.X_LL_Scaled) + self.X_LL_Scaled
        #scaled = z* (self.X_UL_Scaled - self.X_LL_Scaled) + self.X_LL_Scaled

        '''
        x = self.normalizer.inverse_transform(x)
        
        #scaled = x* (self.X_UL_norm - self.X_LL_norm) + self.X_LL_norm
        '''
        #z = (z + 1.0) * 0.5 * (8.0) + 4.0
       
        #scaled = z*self.X_std + self.X_mean
        #scaled = self.normalizer.inverse_transform(scaled)
        return scaled     
    
'''
=================================================================
Classifier and Regression Classes
=================================================================
'''
# First Step: make a classifier object:
class Classifier_Model(torch.nn.Module):
    def __init__(self, Dict):
        nn.Module.__init__(self)
        
        self.xdim = Dict['xdim']
        self.tdim = Dict['tdim']
        self.cdim = Dict['cdim']

        self.net = Dict['net']
        self.epochs = Dict['Training_Epochs']
        
        self.fc = nn.ModuleList()
        
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.tdim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
        
        self.X_embed = nn.Linear(self.xdim, self.tdim)
        
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        '''
        self.fc.append(self.LinLayer(self.xdim,self.net[0]))
        '''
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i]))
            
        
        self.fc.append(nn.Sequential(nn.Linear(self.net[-1], self.cdim), nn.Sigmoid()))

    def LinLayer(self, dimi, dimo):
        
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             #nn.BatchNorm1d(dimo),
                             nn.Dropout(p=0.1))
        

    def forward(self, x):
        
        x = self.X_embed(x)
        
        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
    
        
        return x
    
class Regression_ResNet(torch.nn.Module):
    def __init__(self, Reg_Dict):
        nn.Module.__init__(self)
        
        self.xdim = Reg_Dict['xdim']
        self.ydim = 1
        self.tdim = Reg_Dict['tdim']
        self.net = Reg_Dict['net']
        
        self.fc = nn.ModuleList()
        
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i]))
            
        self.fc.append(self.LinLayer(self.net[-1], self.tdim))
        '''
        #self.tc = nn.ModuleList()

        #for i in range(0, len(self.net)):
            self.tc.append(self.LinLayer(self.tdim,self.net[i]))
        self.tc.append(self.LinLayer(self.tdim, self.tdim))
        '''
        self.finalLayer = nn.Sequential(nn.Linear(self.tdim, self.ydim))
        
    
        self.X_embed = nn.Linear(self.xdim, self.tdim)
        #self.T_embed = nn.Linear(self.ydim, self.tdim)
       
        
    def LinLayer(self, dimi, dimo):
        
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             nn.LayerNorm(dimo),
                             nn.Dropout(p=0.1))
    
    def forward(self, x):
        x = self.X_embed(x)
    
        res_x = x

        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
        
        x = torch.add(x,res_x)
        x = self.finalLayer(x)
        
        return x
    

class Drag_Regression_ResNet(torch.nn.Module):
    def __init__(self, Reg_Dict):
        nn.Module.__init__(self)
        
        self.xdim = Reg_Dict['xdim']+3 # Add 3 Draft, Velocity (Froude Number), and Length scale (LOA)
        self.ydim = 1
        self.tdim = Reg_Dict['tdim']
        self.net = Reg_Dict['net']
        
        self.fc = nn.ModuleList()
        
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i]))
            
        self.fc.append(self.LinLayer(self.net[-1], self.tdim))
        '''
        #self.tc = nn.ModuleList()

        #for i in range(0, len(self.net)):
            self.tc.append(self.LinLayer(self.tdim,self.net[i]))
        self.tc.append(self.LinLayer(self.tdim, self.tdim))
        '''
        self.finalLayer = nn.Sequential(nn.Linear(self.tdim, self.ydim))
        
    
        self.X_embed = nn.Linear(self.xdim, self.tdim)
        #self.T_embed = nn.Linear(self.ydim, self.tdim)
       
        
    def LinLayer(self, dimi, dimo):
        
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             nn.LayerNorm(dimo),
                             nn.Dropout(p=0.1))
    
    def forward(self, x):
        x = self.X_embed(x)
    
        res_x = x

        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
        
        x = torch.add(x,res_x)
        x = self.finalLayer(x)
        
        return x





'''
=================================================================
Diffusion Functions
=================================================================
'''

def timestep_embedding(timesteps, dim, max_period=10000, device=torch.device('cuda:0')):
    """
    From https://github.com/rotot0/tab-ddpm
    
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1,device=device)
    return embedding

def generate_performance_weights(num_samples,num_metrics, gen_type='random'):
    
    weights = np.zeros((num_samples,num_metrics))
     
    if gen_type == 'random':
        for i in range(0,num_samples):
            a = np.random.rand(1,num_metrics)
            weights[i] = a/np.sum(a)
            
    elif gen_type == 'uniform':
        samples = []
        
        steps = np.linspace(0.0,1.0,11)
        
        for i in range(0, len(steps)):
            for j in range(0,len(steps)-i):
                samples.append([steps[i],steps[j],1.0-steps[i]-steps[j]])
        samples = np.array(samples)
        
        L = len(samples)
        
        print(L)
        
        A = np.random.randint(0,L,num_samples)
        
        for i in range(0,num_samples):
            weights[i] = samples[A[i]]

    return weights



# Now lets make a Denoise Model:
    
class Denoise_ResNet_Model(torch.nn.Module):
    def __init__(self, DDPM_Dict):
        nn.Module.__init__(self)
        
        self.xdim = DDPM_Dict['xdim']
        self.ydim = DDPM_Dict['ydim']
        self.tdim  = DDPM_Dict['tdim']
        self.cdim = DDPM_Dict['cdim']
        self.net = DDPM_Dict['net']
        
        self.fc = nn.ModuleList()
        
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i]))
            
        self.fc.append(self.LinLayer(self.net[-1], self.tdim))
        
        
        self.finalLayer = nn.Sequential(nn.Linear(self.tdim, self.xdim))
        
    
        self.X_embed = nn.Linear(self.xdim, self.tdim)
        

        self.Con_embed = nn.Sequential(
                    nn.Linear(self.cdim, self.tdim),
                    nn.SiLU(),
                    nn.Linear(self.tdim, self.tdim))
        

        
        self.time_embed = nn.Sequential(
            nn.Linear(self.tdim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
       
        
    def LinLayer(self, dimi, dimo):
        
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             nn.BatchNorm1d(dimo),
                             nn.Dropout(p=0.1))
        


    def forward(self, x, cons, timesteps):
        
                
        x = self.X_embed(x) + self.time_embed(timestep_embedding(timesteps, self.tdim)) + self.Con_embed(cons)
        res_x = x
        
        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
    
        x = torch.add(x,res_x)
        
        x = self.finalLayer(x)
        
        return x



    
'''
==============================================================================
EMA - Exponential Moving Average: Helps with stable training
========================================================================
EMA class from: https://github.com/azad-academy/denoising-diffusion-model/blob/main/ema.py

'''
# Exponential Moving Average Class
# Orignal source: https://github.com/acids-ircam/diffusion_models


class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
        
    
        
'''
=======================================================================
Trainer class modified from Tab-ddpm paper code with help from hugging face
=====================================================================
'''
class GuidedDiffusionEnv:
    #def __init__(self, DDPM_Dict, Class_Dict, Reg_Dict, X,):
    def __init__(self, DDPM_Dict, Class_Dict, Drag_Reg_Dict,LOA_wBulb_Reg_Dict, WL_Reg_Dict,Vol_Reg_Dict, X, X_neg, VolVec, BOAVec, DdVec):   
        self.DDPM_Dict = DDPM_Dict
        self.datalength = self.DDPM_Dict['datalength']
        self.batch_size = self.DDPM_Dict['batch_size']  
        self.Class_Dict = Class_Dict
        self.Drag_Reg_Dict = Drag_Reg_Dict
        self.LOA_wBulb_Reg_Dict = LOA_wBulb_Reg_Dict
        self.WL_Reg_Dict = WL_Reg_Dict
        self.Vol_Reg_Dict = Vol_Reg_Dict
                
        
        self.device =torch.device(self.DDPM_Dict['device_name'])
        
        #Build the Diffusion Network
        self.diffusion = Denoise_ResNet_Model(self.DDPM_Dict)


        #Build Classifier Network
        self.classifier = Classifier_Model(self.Class_Dict)

        #Build Regression Networks:
        #self.load_trained_Drag_regressor()


        #self.num_regressors = self.Reg_Dict['num_regressors']
        #self.load_trained_regressors()
        
        self.diffusion.to(self.device)
        self.classifier.to(self.device)

        self.gamma = self.DDPM_Dict['gamma']
        self.lam = self.DDPM_Dict['lambda']
        

        '''
        for i in range(0,self.num_regressors):
            self.regressors[i].to(self.device)

        self.dataLength = self.DDPM_Dict['datalength']
        self.batch_size = self.DDPM_Dict['batch_size']
        
        self.lambdas = np.array(self.DDPM_Dict['lambdas'])
        
        '''
        self.data_norm = Data_Normalizer(np.array(self.DDPM_Dict['X_LL']),np.array(self.DDPM_Dict['X_UL']),self.datalength)
    
        # Set Up Design Data
        self.X = self.data_norm.fit_Data(X)
        self.X = torch.from_numpy(self.X.astype('float32'))
        
        #Set Up Negative Design Data
        self.X_neg = self.data_norm.transform_Data(X_neg)
        self.X_neg = torch.from_numpy(self.X_neg.astype('float32'))

        #Set Up Feasibility Labels
        self.Cons = torch.from_numpy(np.zeros((len(self.X),1)).astype('float32'))
        self.Cons_neg = torch.from_numpy(np.ones((len(self.X_neg),1)).astype('float32'))
        
        self.num_WL_Steps = len(VolVec[0]) #should be 101
       

        self.T_range = [.25,.67]

        self.T_vec = np.linspace(0,1,self.num_WL_Steps)
        self.VolVec = VolVec
        self.BOAVec = BOAVec
        self.DdVec = DdVec
        


        '''
        self.X_neg = self.data_norm.transform_Data(X_neg)
        
        #X and Y are numpy arrays - convert to tensors
        self.X_neg = torch.from_numpy(self.X_neg.astype('float32'))
        self.Y = torch.from_numpy(Y.astype('float32'))
        
        self.Cons = torch.from_numpy(Cons.astype('float32'))
        
        self.Cons_neg = torch.from_numpy(Cons_neg.astype('float32'))

        
        
        self.X_neg = self.X_neg.to(self.device)
        self.Y = self.Y.to(self.device)
        '''

        self.X = self.X.to(self.device)
        self.X_neg = self.X_neg.to(self.device)
        self.Cons = self.Cons.to(self.device)
        self.Cons_neg = self.Cons_neg.to(self.device)
        

        self.eps = 1e-8
        
        self.ema = EMA(0.99)
        self.ema.register(self.diffusion)
        
        
        #set up optimizer 
        self.timesteps = self.DDPM_Dict['Diffusion_Timesteps']
        self.num_diffusion_epochs = self.DDPM_Dict['Training_Epochs']
        
        #self.num_classifier_epochs = self.Class_Dict['Training_Epochs']
        #self.num_regressor_epochs = self.Reg_Dict['Training_Epochs']
        
        lr = self.DDPM_Dict['lr']
        self.init_lr = lr
        weight_decay = self.DDPM_Dict['weight_decay']
        
        self.optimizer_diffusion = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer_classifier = torch.optim.AdamW(self.classifier.parameters(),lr=.001, weight_decay=weight_decay)
        #self.optimizer_regressors = [torch.optim.AdamW(self.regressors[i].parameters(),lr=.001, weight_decay=weight_decay) for i in range(0,self.Reg_Dict['num_regressors'])]
        


        self.log_every = 100
        self.print_every = 5000
        self.loss_history = []
        
        
        
        #Set up alpha terms
        self.betas = torch.linspace(0.001, 0.2, self.timesteps).to(self.device)
        
        #self.betas = betas_for_alpha_bar(self.timesteps, lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,)
        #self.betas = torch.from_numpy(self.betas.astype('float32')).to(self.device)
        
        self.alphas = 1. - self.betas
        
        self.log_alpha = torch.log(self.alphas)
        self.log_cumprod_alpha = np.cumsum(self.log_alpha.cpu().numpy())
        
        self.log_cumprod_alpha = torch.tensor(self.log_cumprod_alpha,device=self.device)
        
        
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1],[1,0],'constant', 0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod =  torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        a = torch.clone(self.posterior_variance)
        a[0] = a[1]                 
        self.posterior_log_variance_clipped = torch.log(a)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev)* torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

    """++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Start the training model functions
    """
    def extract(self,a, t, x_shape):
        b, *_ = t.shape
        t = t.to(a.device)
        out = a.gather(-1, t)
        while len(out.shape) < len(x_shape):
            out = out[..., None]
        return out.expand(x_shape)
    
    def _anneal_lr(self, epoch_step):
        #Update the learning rate
        frac_done = epoch_step / self.num_diffusion_epochs
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer_diffusion.param_groups:
            param_group["lr"] = lr
    
    def interp(self,A,Z,z):
        # This function interpolates data to approximate A(z)  given values of A(Z) 
    
        idx = np.where(Z < z)[0][-1]
    
        frac = (z - Z[idx])/(Z[idx+1] - Z[idx])
    
        return A[idx] + frac*(A[idx+1] - A[idx])

    def batch_train(self, batch_size=None):
        '''
        This function takes in a batch of design vectors and outputs the corresponding CT values
        '''
        if batch_size == None:
            batch_size = self.batch_size
            
        A = A = np.random.randint(0,self.datalength,batch_size)
        #Random Waterline

        t = np.random.uniform(self.T_range[0], self.T_range[1], (batch_size,))
        t_tens = torch.tensor(t[:,np.newaxis]).float().to(self.device)


        #interp volume for conditioning
        Vol = np.array([self.interp(self.VolVec[A[i]],self.T_vec,t[i]) for i in range(0,len(t))]) #Non-dimensionalized Waterline Length    
        Dd = self.DdVec[A]  
        BOA = self.BOAVec[A]
        
        x_batch = self.X[A]
        
        cond_batch = np.concatenate((t[:,np.newaxis],BOA[:,np.newaxis],Dd[:,np.newaxis],Vol[:,np.newaxis]),axis=1)

        cond_batch = torch.from_numpy(cond_batch.astype('float32')).to(self.device)
        
        return x_batch, cond_batch
               
            
    '''
    =========================================================================
    Vanilla Diffusion
    ==========================================================================
    '''
    def q_sample(self,x_start, t, noise=None):
        """
        qsample from https://huggingface.co/blog/annotated-diffusion
        """
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
    
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
    
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    
    def p_loss(self,x_start, cond, t, noise=None,loss_type='l2'):
        '''
        from https://huggingface.co/blog/annotated-diffusion
        '''
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.diffusion(x_noisy, cond, t)
        
        #predicted_noise = predicted_noise.clamp(-3,3)
        
        if loss_type == 'l1':
            loss1 = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss1 = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss1 = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
                
        return loss1 
        



    '''
    ==============================================================================
    Diffusion Training and Sampling Functions
    ==============================================================================
    '''      


    def run_diffusion_step(self, x,cond):
        self.optimizer_diffusion.zero_grad()
        
        t = torch.randint(0,self.timesteps,(self.batch_size,),device=self.device)
        loss1 = self.p_loss(x,cond, t,loss_type='l2')
        
        loss = loss1 
        loss.backward()
        self.optimizer_diffusion.step()

        return loss

    def run_train_diffusion_loop(self, batches_per_epoch=100):
        print('Denoising Model Training...')
        self.diffusion.train()
        
        num_batches = self.datalength // self.batch_size
        
        batches_per_epoch = min(num_batches,batches_per_epoch)
        
        
        
        for i in tqdm(range(self.num_diffusion_epochs)):
            
            #IDX = permute_idx(self.dataLength) # get randomized list of idx for batching
            
            for j in range(0,batches_per_epoch):
                
                x_batch, cond_batch = self.batch_train()
                       
                loss = self.run_diffusion_step(x_batch, cond_batch)
                '''
                
                Gaussian Diffusion (oooohhhh ahhhhhh) from TabDDPM:
                '''
                #loss = self.train_step(x_batch[j])

            self._anneal_lr(i)

            if (i + 1) % self.log_every == 0:
                self.loss_history.append([i+1,float(loss.to('cpu').detach().numpy())])
        
            if (i + 1) % self.print_every == 0:
                    print(f'Step {(i + 1)}/{self.num_diffusion_epochs} Loss: {loss}')
                

            self.ema.update(self.diffusion)
        #Make Loss History an np array
        self.loss_history = np.array(self.loss_history)
        print('Denoising Model Training Complete!')
    
    def fease_fn(self, x):
        #From OpenAI: https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py
        
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            
            pred_cons = self.classifier(x_in)
            
            
            
            #log_p = torch.log(pred_cons)
            
            #sign = torch.sign(cons-0.5)

            grad = torch.autograd.grad(pred_cons.sum(), x_in)[0] 
            
            #print(grad[0])          
            return -grad
        
    def Vol_fn(self, x, cond):
        #From OpenAI: https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py
        x_in = torch.cat((x,cond[:,0:1]),dim=1)
        with torch.enable_grad():
            x_in = x_in.detach().requires_grad_(True)

            
            
            pred_vol = self.Vol_Reg(x_in)
            
            
            
            #log_p = torch.log(pred_cons)
            
            #sign = torch.sign(cons-0.5)

            grad = -2.0*(cond[:,3:4] - pred_vol)*torch.autograd.grad(pred_vol.sum(), x_in)[0] 
            
            #print(grad[0])          
            return grad[:,0:len(x[0])]
        
    def Drag_fn(self, x, drag_cond, g=9.81):
        #From OpenAI: https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py
        # drag_cond = [ToD, Fn, LOA]
        #concatenate the drag conditioning to the design vector

        #LOA = drag_cond[:,2:3]/ self.LOA_wBulb_Reg(x) #Calculate the desired LOA considering the bulbs
        LOA = drag_cond[:,2:3]

        x_in = torch.cat((x,drag_cond[:,0:1]),dim=1) #concatenate ToD for WL Prediction

        Fn_cond = drag_cond[:,1:2]/torch.sqrt(g* LOA*self.WL_Reg(x_in)) #Calculate Froude Number for embedding

        #print(x.shape)
        #print(Fn_cond.shape)
        #print(LOA.shape)

        x_in = torch.cat((x_in,Fn_cond, torch.log10(LOA)),dim=1) #concatenate for drag prediction 



        with torch.enable_grad():
            x_in = x_in.detach().requires_grad_(True)

            
            
            perf = self.Drag_Reg(x_in)
            

            grad = torch.autograd.grad(perf.sum(), x_in)[0] 
            
            #print(grad[0])          
            return grad[:,0:len(x[0])]
        
    @torch.no_grad()
    def p_sample(self, x, t, cons):
        
        time= torch.full((x.size(dim=0),),t,dtype=torch.int64,device=self.device)
        
        X_diff = self.diffusion(x, cons, time) 
        
        
        betas_t = self.extract(self.betas, time, x.shape)
        
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, time, x.shape
        )
        
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, time, x.shape)
        
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
    
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * X_diff/ sqrt_one_minus_alphas_cumprod_t
        )
        
        
        posterior_variance_t = self.extract(self.posterior_variance, time, x.shape)
        
        
        fease_grad = self.fease_fn(x)
        #print(gradient.detach().to('cpu')[0])

        if t == 0:
            return model_mean
        else:
            
            noise = torch.randn_like(x,device=self.device)
            # Dot product gradient to noise
            return model_mean + torch.sqrt(posterior_variance_t) * (noise*(1.0-self.gamma) + self.gamma*fease_grad.float())
            #return model_mean + torch.sqrt(posterior_variance_t) * (noise + self.gamma*fease_grad.float())
       
    @torch.no_grad()
    def drag_p_sample(self, x, t, geom_cons, drag_cons):
        
        time= torch.full((x.size(dim=0),),t,dtype=torch.int64,device=self.device)
        
        X_diff = self.diffusion(x, geom_cons, time) 
        
        
        betas_t = self.extract(self.betas, time, x.shape)
        
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, time, x.shape
        )
        
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, time, x.shape)
        
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
    
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * X_diff/ sqrt_one_minus_alphas_cumprod_t
        )
        
        
        posterior_variance_t = self.extract(self.posterior_variance, time, x.shape)
        
        
        fease_grad = self.fease_fn(x)

        drag_grad = self.Drag_fn(x,drag_cons)
        #print(gradient.detach().to('cpu')[0])

        if t == 0:
            return model_mean
        else:
            
            noise = torch.randn_like(x,device=self.device)
            # Dot product gradient to noise
            return model_mean + torch.sqrt(posterior_variance_t) * (noise*(1.0-self.gamma) + self.gamma*fease_grad.float() - self.lam[0]*drag_grad.float())
            #return model_mean + torch.sqrt(posterior_variance_t) * (noise + self.gamma*fease_grad.float() - self.lam*drag_grad.float())
             
    @torch.no_grad()
    def vol_drag_p_sample(self, x, t, geom_cons, drag_cons):
        
        time= torch.full((x.size(dim=0),),t,dtype=torch.int64,device=self.device)
        
        X_diff = self.diffusion(x, geom_cons, time) 
        
        
        betas_t = self.extract(self.betas, time, x.shape)
        
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, time, x.shape
        )
        
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, time, x.shape)
        
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
    
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * X_diff/ sqrt_one_minus_alphas_cumprod_t
        )
        
        
        posterior_variance_t = self.extract(self.posterior_variance, time, x.shape)
        
        
        fease_grad = self.fease_fn(x)

        drag_grad = self.Drag_fn(x,drag_cons)

        vol_grad = self.Vol_fn(x,geom_cons)
        #print(gradient.detach().to('cpu')[0])

        if t == 0:
            return model_mean
        else:
            
            noise = torch.randn_like(x,device=self.device)
            # Dot product gradient to noise
            #return model_mean + torch.sqrt(posterior_variance_t) * (noise*(1.0-self.gamma) + self.gamma*fease_grad.float() - self.lam[0]*drag_grad.float() - self.lam[1]*vol_grad.float())
            return model_mean + torch.sqrt(posterior_variance_t) * (noise*(1.0-self.gamma) + self.gamma*fease_grad.float()) - self.lam[0]*drag_grad.float() - self.lam[1]*vol_grad.float()
            
        
    @torch.no_grad()
    def gen_cond_samples(self, cons):
        #COND is a numpy array of the conditioning it is shape (num_samples,conditioning terms)
        num_samples = len(cons)

        cons = torch.from_numpy(cons.astype('float32'))
        cons = cons.to(self.device)
        
        #print(num_samples) #should be 1
        
        x_gen = torch.randn((num_samples,self.diffusion.xdim),device=self.device)
        
        self.diffusion.eval()
        self.classifier.eval()
        
   
        for i in tqdm(range(self.timesteps - 1, 0, -1)):


            x_gen = self.p_sample(x_gen, i,cons)

        
        output = x_gen.cpu().detach().numpy()
            
            
        output_scaled = self.data_norm.scale_X(output)
        
        return output_scaled, output
    
    def gen_low_drag_samples(self, Geom_COND, Drag_COND):
        #Geom_COND is a numpy array of the geometric conditioning. Each row is [ToD, BoL, DoL, Vol]
        #Drag_COND is a numpy array of the drag conditioning. Each Row is [ToD, U [m/s], LOA [m]]
        num_samples = len(Geom_COND)
        
        geom_cons = torch.from_numpy(Geom_COND.astype('float32'))
        geom_cons = geom_cons.to(self.device)

        drag_cons = torch.from_numpy(Drag_COND.astype('float32'))
        drag_cons = drag_cons.to(self.device)
        
        #print(num_samples) #should be 1
        
        x_gen = torch.randn((num_samples,self.diffusion.xdim),device=self.device)
        
        self.diffusion.eval()
        self.classifier.eval()



        guidance_step = 16
   
        for i in tqdm(range(self.timesteps - 1, guidance_step, -1)):


            x_gen = self.drag_p_sample(x_gen, i,geom_cons, drag_cons)
        
        for i in tqdm(range(guidance_step, 0, -1)):


            x_gen = self.p_sample(x_gen, i,geom_cons)

        
        output = x_gen.cpu().detach().numpy()
            
            
        output_scaled = self.data_norm.scale_X(output)

        #LOA = drag_cons[:,2:3]/ self.LOA_wBulb_Reg(x_gen) #Calculate the desired LOA considering the bulbs
        OA = drag_cons[:,2:3]
        LOA = LOA.cpu().detach().numpy()

        output_scaled = np.concatenate((LOA,output_scaled),axis=1)
        
        return output_scaled, output
    
    def gen_vol_drag_guided_samples(self, Geom_COND, Drag_COND):
        #Geom_COND is a numpy array of the geometric conditioning. Each row is [ToD, BoL, DoL, Vol]
        #Drag_COND is a numpy array of the drag conditioning. Each Row is [ToD, U [m/s], LOA [m]]
        num_samples = len(Geom_COND)
        
        geom_cons = torch.from_numpy(Geom_COND.astype('float32'))
        geom_cons = geom_cons.to(self.device)

        drag_cons = torch.from_numpy(Drag_COND.astype('float32'))
        drag_cons = drag_cons.to(self.device)
        
        #print(num_samples) #should be 1
        
        x_gen = torch.randn((num_samples,self.diffusion.xdim),device=self.device)
        
        self.diffusion.eval()
        self.classifier.eval()
        
        guidance_step = 32
   
        for i in tqdm(range(self.timesteps - 1, guidance_step, -1)):


            x_gen = self.vol_drag_p_sample(x_gen, i,geom_cons, drag_cons)
        
        for i in tqdm(range(guidance_step, 0, -1)):


            x_gen = self.p_sample(x_gen, i,geom_cons)

        
        output = x_gen.cpu().detach().numpy()
            
            
        output_scaled = self.data_norm.scale_X(output)

        #LOA = drag_cons[:,2:3]/ self.LOA_wBulb_Reg(x_gen) #Calculate the desired LOA considering the bulbs
        LOA = drag_cons[:,2:3]

        LOA = LOA.cpu().detach().numpy()

        output_scaled = np.concatenate((LOA,output_scaled),axis=1)
        
        return output_scaled, output
    
   
    '''
    ==============================================================================
    Classifier and Regression Training Functions
    ==============================================================================
    '''
           
    
    def run_classifier_step(self,x,cons):
        
        self.optimizer_classifier.zero_grad()
        

        predicted_cons = self.classifier(x)
        
        loss = F.binary_cross_entropy(predicted_cons, cons) #F.mse_loss(predicted_cons, cons) #F.binary_cross_entropy(predicted_cons, cons)
        loss.backward()
        self.optimizer_classifier.step()
        
        return loss
    
    def run_train_classifier_loop(self, batches_per_epoch=100):
        
        X = torch.cat((self.X,self.X_neg))
        C = torch.cat((self.Cons,self.Cons_neg))

        test = np.random.randint(0,len(X),int(len(X)*.25))

        X_train = X[~test]
        C_train = C[~test]  

        X_test = X[test]
        C_test = C[test]
        
        print(C_train.shape)
        
        datalength = X_train.shape[0]
        
        print('Classifier Model Training...')
        self.classifier.train()
        
        num_batches = datalength // self.batch_size
        
        batches_per_epoch = min(num_batches,batches_per_epoch)
    
        
        for i in tqdm(range(self.classifier.epochs)):
            
            #IDX = permute_idx(self.dataLength) # get randomized list of idx for batching
            
            for j in range(0,batches_per_epoch):
                
                A = np.random.randint(0,datalength,self.batch_size)
                x_batch = X_train[A] 
                #y_batch[j] = self.Y[IDX[j*self.batch_size:(j+1)*self.batch_size]] 
                cons_batch = C_train[A]
                #cons_batch[j] = self.Cons[IDX[j*self.batch_size:(j+1)*self.batch_size]]
                       
                loss = self.run_classifier_step(x_batch,cons_batch)    
        '''    
        for i in tqdm(range(0,self.num_classifier_epochs)):
            loss = self.run_classifier_step(X,C)  
        '''   
        self.classifier.eval()
        
        
        
        
        C_pred = self.classifier(X_test)
        

        C_pred = C_pred.to(torch.device('cpu')).detach().numpy()
       
        #print(C_pred.shape)
        C_pred = np.rint(C_pred) #Make it an iteger guess

        C_test = C_test.to(torch.device('cpu')).detach().numpy()
      
        F1 = f1_score(C_test,C_pred)

        print('F1 score: ' + str(F1))
        
        print('Classifier Training Complete!')

    def Predict_Drag(self,x,drag_cond, rho=1025.0, g=9.81):
        #x is a numpy array of the normalized design vector (alread)
        #drag_cond is a numpy array of the drag conditioning. Each Row is [ToD, U[m/s], LOA]
      
        x = torch.from_numpy(x.astype('float32')).to(self.device)

        drag_cond = torch.from_numpy(drag_cond.astype('float32')).to(self.device)

        #LOA = drag_cond[:,2:3]/ self.LOA_wBulb_Reg(x) #Calculate the desired LOA considering the bulbs

        LOA = drag_cond[:,2:3]
        
        x = torch.cat((x,drag_cond[:,0:1]),dim=1) #concatenate ToD for WL Prediction

        Fn_cond = drag_cond[:,1:2]/torch.sqrt(g* LOA*self.WL_Reg(x)) #Calculate Froude Number for embedding
        
        x = torch.cat((x,Fn_cond, torch.log10(LOA)),dim=1) #concatenate for drag prediction 
        
        self.Drag_Reg.eval()
        
        CT = 10**self.Drag_Reg(x)

        Drag = CT*0.5*rho*(drag_cond[:,1:2]**2)*LOA**2 #Calculate Drag: RT = CT*0.5*rho*U^2*LOA^2
        
        Drag = Drag.to(torch.device('cpu')).detach().numpy()
        
        return Drag


    '''
    ==============================================================================
    Saving and Loading Model Functions
    ==============================================================================
    '''

    def load_trained_diffusion_model(self,PATH):
        #PATH is full path to the state dictionary, including the file name and extension
        self.diffusion.load_state_dict(torch.load(PATH))
    
    def Load_Dict(PATH):
        #returns the dictionary for the DDPM_Dictionary to rebuild the model
        #PATH is the path including file name and extension of the json file that stores it. 
        f = open(PATH)
        return json.loads(f)
        
    
    def Save_diffusion_model(self,PATH,name):
        '''
        PATH is the path to the folder to store this in, including '/' at the end
        name is the name of the model to save without an extension
        '''
        torch.save(self.diffusion.state_dict(), PATH+name+'_diffusion.pth')
        
        JSON = json.dumps(self.DDPM_Dict)
        f = open(PATH+name+'.json', 'w')
        f.write(JSON)
        f.close()
        
    def load_trained_classifier_model(self,PATH):
        #PATH is full path to the state dictionary, including the file name and extension
        self.classifier.load_state_dict(torch.load(PATH))
        
    def load_trained_Drag_regression_models(self,PATH):
        #label = self.Drag_Reg_Dict['Model_Paths']
        
        self.Drag_Reg = Drag_Regression_ResNet(self.Drag_Reg_Dict)
        self.Drag_Reg.load_state_dict(torch.load(PATH[0]))
        self.Drag_Reg.to(self.device)

        self.LOA_wBulb_Reg = Regression_ResNet(self.LOA_wBulb_Reg_Dict)
        self.LOA_wBulb_Reg.load_state_dict(torch.load(PATH[1]))
        self.LOA_wBulb_Reg.to(self.device)
        
        self.WL_Reg = Regression_ResNet(self.WL_Reg_Dict)
        self.WL_Reg.load_state_dict(torch.load(PATH[2]))
        self.WL_Reg.to(self.device)

        self.Vol_Reg = Regression_ResNet(self.Vol_Reg_Dict)
        self.Vol_Reg.load_state_dict(torch.load(PATH[3]))
        self.Vol_Reg.to(self.device)
        self.Drag_Reg.eval()
        self.LOA_wBulb_Reg.eval()
        self.WL_Reg.eval()
        self.Vol_Reg.eval()




    
    
    def Save_classifier_model(self,PATH,name):
        '''
        PATH is the path to the folder to store this in, including '/' at the end
        name is the name of the model to save without an extension
        '''
        torch.save(self.classifier.state_dict(), PATH+name+ '.pth')
        
        JSON = json.dumps(self.Class_Dict)
        f = open(PATH+name+ '.json', 'w')
        f.write(JSON)
        f.close()
    
    def Save_regression_models(self,PATH):
        '''
        PATH is the path to the folder to store this in, including '/' at the end
       
        '''
        for i in range(0,len(self.regressors)):
            torch.save(self.regressors[i].state_dict(), PATH + self.Reg_Dict['Model_Labels'][i] +'.pth')
        
        JSON = json.dumps(self.Reg_Dict)
        f = open(PATH + '_regressor_Dict.json', 'w')
        f.write(JSON)
        f.close()
