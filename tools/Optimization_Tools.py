import sys
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import Guided_Cond_DDPM_Tools as GC_DDPM
import sklearn.preprocessing as PP

sys.path.append('/home/ada/Documents/HullParameterization')
from HullParameterization import Hull_Parameterization as HP


'''
===================================================================================================
Design Vector Tools
===================================================================================================
'''

# Build the tools
idx_Bits = np.array([20,21,31,32])
idx_parabola_scaling = np.array([12,16,18,26])

idx_BBFactors = [33,34,35,36,37]
idx_BB = 31

idx_SBFactors = [38,39,40,41,42,43,44]
idx_SB = 32

def clean_designVector(DesVec):
    for i in range(0,len(DesVec)):

        DesVec[i,idx_BB] = (DesVec[i,idx_BB] + 0.5) // 1 #int rounds to 1 or 0
        DesVec[i,idx_SB] = (DesVec[i,idx_SB] + 0.5) // 1 #int rounds to 1 or 0

        if sum(abs(DesVec[i,idx_BBFactors]) < 1e-6) > 0:
            DesVec[i,idx_BB] = 0.
        if sum(abs(DesVec[i,idx_SBFactors]) < 1e-6) > 0:
            DesVec[i,idx_SB] = 0.
           
    
        DesVec[i,idx_BBFactors] = DesVec[i,idx_BB] * DesVec[i,idx_BBFactors] 
        DesVec[i,idx_SBFactors] = DesVec[i,idx_SB] * DesVec[i,idx_SBFactors]

    return DesVec






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
===================================================================================================
Neural Network Tools
===================================================================================================
'''

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

class Drag_Pred_Env:
    
    def __init__(self, MODEL_PATHS, X_Limits, X_ds, Drag_Reg_Dict, LOA_wBulb_Reg_Dict, WL_Reg_Dict, device_name):
        self.Drag_Reg_Dict = Drag_Reg_Dict
        self.LOA_wBulb_Reg_Dict = LOA_wBulb_Reg_Dict
        self.WL_Reg_Dict = WL_Reg_Dict

        self.device =torch.device(device_name)
      

        self.load_trained_Drag_regression_models(MODEL_PATHS)

        self.datalength = len(X_ds)

        self.data_norm = Data_Normalizer(X_Limits[1:,0], X_Limits[1:,1],self.datalength) #only want to normalize the 44 design params, not loa
        a = self.data_norm.fit_Data(X_ds)

    def load_trained_Drag_regression_models(self, PATH):
        #label = self.Drag_Reg_Dict['Model_Paths']
        
        self.Drag_Reg = GC_DDPM.Drag_Regression_ResNet(self.Drag_Reg_Dict)
        self.Drag_Reg.load_state_dict(torch.load(PATH[0]))
        self.Drag_Reg.to(self.device)

        self.LOA_wBulb_Reg = GC_DDPM.Regression_ResNet(self.LOA_wBulb_Reg_Dict)
        self.LOA_wBulb_Reg.load_state_dict(torch.load(PATH[1]))
        self.LOA_wBulb_Reg.to(self.device)
        
        self.WL_Reg = GC_DDPM.Regression_ResNet(self.WL_Reg_Dict)
        self.WL_Reg.load_state_dict(torch.load(PATH[2]))
        self.WL_Reg.to(self.device)

    def Predict_Drag(self,x,drag_cond, LOA_adj, rho=1025.0, g=9.81):
        #x is a numpy array of the normalized design vector (alread)
        #drag_cond is a numpy array of the drag conditioning. Each Row is [ToD, U[m/s], LOA]
        
        x = torch.from_numpy(x.astype('float32')).to(self.device)


        drag_cond = torch.from_numpy(drag_cond.astype('float32')).to(self.device)

        LOA_adj = torch.from_numpy(LOA_adj.astype('float32')).to(self.device)
        #LOA = drag_cond[:,2:3]/ self.LOA_wBulb_Reg(x) #Calculate the desired LOA considering the bulbs
        #LOA = drag_cond[:,2:3]
        
        x = torch.cat((x,drag_cond[:,0:1]),dim=1) #concatenate ToD for WL Prediction

        Fn_cond = drag_cond[:,1:2]/torch.sqrt(g* LOA_adj*self.WL_Reg(x)) #Calculate Froude Number for embedding
        
        x = torch.cat((x,Fn_cond, torch.log10(LOA_adj)),dim=1) #concatenate for drag prediction 
        
        self.Drag_Reg.eval()
        
        CT = 10**self.Drag_Reg(x)

        Drag = CT*0.5*rho*(drag_cond[:,1:2]**2)*LOA_adj**2 #Calculate Drag: RT = CT*0.5*rho*U^2*LOA^2
        
        Drag = Drag.to(torch.device('cpu')).detach().numpy()
        
        return Drag, CT.to(torch.device('cpu')).detach().numpy(), Fn_cond.to(torch.device('cpu')).detach().numpy()

'''
===================================================================================================
Optimization Problem Tools
===================================================================================================
'''
from pymoo.core.problem import Problem
# Build the minimization problem:

class Minimize_Drag(Problem):

    def __init__(self, Geom_COND, MODEL_PATHS, X_Limits, X, Drag_Reg_Dict, LOA_wBulb_Reg_Dict, WL_Reg_Dict, device_name):
        
        #Establish Upper and Lower Limmits on design vector
        self.num_cons = 49+8
        self.D = Drag_Pred_Env(MODEL_PATHS, X_Limits, X, Drag_Reg_Dict, LOA_wBulb_Reg_Dict, WL_Reg_Dict, device_name)

        self.LOA_target = Geom_COND[0]
        self.BOA_target = Geom_COND[1]
        self.DRAFT_target = Geom_COND[2]
        self.DEPTH_target = Geom_COND[3]
        self.Vol_target = Geom_COND[4]
        self.U = Geom_COND[5]
        self.nGen = Geom_COND[6]
        self.CB = self.Vol_target/(self.LOA_target*self.BOA_target*self.DRAFT_target)
        self.Gen_count = 0

        self.dim = np.array([[self.DRAFT_target/self.DEPTH_target, self.U, self.LOA_target]]) #Drag_conditioning is [ToD, U(m/s), LOA (m)]
        
        BoL = self.BOA_target/self.LOA_target

        X_Limits[3] = [0.97*BoL, 1.05*BoL] # Limit Bd range 
        X_Limits[7] = [0.97*BoL, 1.05*BoL] # Limit Bc range

        super().__init__(n_var=45, n_obj=2, n_constr=self.num_cons, xl=X_Limits[:,0], xu=X_Limits[:,1])
        

    def calc_hulls(self,X):
        DesVec = clean_designVector(X)
        
        cons = np.zeros((len(X),self.num_cons))
        LOA_wBulb = 0
        B = 0
        D = 0
        Vol = 0
        
       

        for i in range(0,len(X)):
            hull = HP(DesVec[i])
            LOA_wBulb = hull.Calc_LOA_wBulb()
            cons[i,0:49] = hull.input_Constraints()
            D = hull.Dd 
            if sum([b > 0 for b in cons[i,0:49]]) > 0: #check if constraints violated
                #print('input constraints violated')
                B = hull.Calc_Max_Beam_midship()

                cons[i,49:55] = [   LOA_wBulb - self.LOA_target, #LOA upper bound
                                    0.98*self.BOA_target - B, #BOA Lower bound Dynamically modified and reduced
                                    B - 1.02*self.BOA_target, #BOA upper bound
                                    0.99*self.DEPTH_target - D, #DOA Lower bound
                                    D - 1.01*self.DEPTH_target, #DOA upper bound
                                    0.99*self.Vol_target - self.CB*self.DRAFT_target*LOA_wBulb*B] #volume lower bound Modified to not have to calc volume on an invalid hull 
             
            else:
                
                try:
                    Z = hull.Calc_VolumeProperties()
                    Vol = HP.interp(hull.Volumes,Z,self.DRAFT_target)
                    #B = hull.Calc_Max_Beam_midship() 
                    B = max(hull.Calc_Max_Beam_midship(), hull.Calc_Max_Beam_PC())
                    cons[i,49:55] = [LOA_wBulb - self.LOA_target, #LOA upper bound
                                    0.98*self.BOA_target - B, #BOA Lower bound
                                    B - 1.02*self.BOA_target, #BOA upper bound
                                    0.99*self.DEPTH_target - D, #DOA Lower bound
                                    D - 1.01*self.DEPTH_target, #DOA upper bound
                                    0.99*self.Vol_target-Vol] #volume lower bound
                except:
                    cons[i,49:55] = [LOA_wBulb - self.LOA_target, #LOA upper bound
                                    0.98*self.BOA_target - B, #BOA Lower bound
                                    B - 1.02*self.BOA_target, #BOA upper bound
                                    0.99*self.DEPTH_target - D, #DOA Lower bound
                                    D - 1.01*self.DEPTH_target, #DOA upper bound
                                    0.01*self.Vol_target] #make the volume calc be violated

        return cons


    def _evaluate(self, X, out, *args, **kwargs):
        
        self.Gen_count += 1
        cons = self.calc_hulls(X)
        #Drag Eval      
        des = clean_designVector(X)
        x_norm = self.D.data_norm.transform_Data(des[:,1:])
        drag_cond = np.repeat(self.dim, len(des), axis=0) #reapeat 
        Rt,Ct, Fn = self.D.Predict_Drag(x_norm,drag_cond, des[:,0:1])

        #Fn constraints
        for i in range(0,len(X)):
 
            cons[i,55:57] = [Fn[i,0] - 0.45, #Fn less than 0.45
                             0.15 - Fn[i,0]] #Fn greater than 0.15

            Rt[i] = Rt[i]*(1+sum(cons[i]>0)*0.1) #penalize drag if constraints violated
            Ct[i] = Ct[i]*(1+sum(cons[i]>0)*0.1) #penalize drag coefficient if constraints violated
        
        #print(self.Gen_count)
        out["F"] = np.concatenate((Rt,Ct),axis=1)
        out["G"] = cons

