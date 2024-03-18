
import torch
import torch.nn as nn
import numpy as np

def Initialize_Parameter_A(dim):
    temp = np.eye(dim)
    lista = [] 
    for i in range(len(temp)):
        lista.append(temp[i,:])
        lista.append(-temp[i,:])
    array_1 = np.array(lista)
    array_1[array_1==-0] = 0

    array_a = np.concatenate((array_1, array_1), axis=0) 
    return array_a

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Transform_Param_A(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, g):
        gs = g.clone()  
        gs = torch.sigmoid(gs) # 
        gs[gs>=sigmoid(1)] = 1
        gs[gs<=sigmoid(-1)] = -1
        gs[(gs>sigmoid(-1))&(gs<sigmoid(1))]=0 
        return gs
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clone(grad_output)
        return grad_input
    
class STEstimator(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, g):
        # g -> gs
        g_clip = torch.clamp(g, min=0, max = 1)  
        gs = g_clip.clone()  
        gs[gs>=0.5] = 1
        gs[gs<0.5] = 0
        return gs
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clone(grad_output)
        return grad_input


class Clip(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, g):
        gs = g.clone()
        gs[gs>0] = 1
        gs[gs<=0] = -1
        return gs
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clone(grad_output)
        return grad_input


class RMinTimeWeight(object):  
    def __init__(self, t1, t2):  
        self.t1 = t1
        self.t2 = t2
        self.tau = torch.tensor(1, requires_grad=False)
        self.relu = nn.ReLU()
    def get_weight(self,w):
        f1 = (self.relu(w-self.t1+self.tau)-self.relu(w-self.t1))/self.tau
        f2 = (self.relu(-w+self.t2+self.tau)-self.relu(-w+self.t2))/self.tau
        w = torch.min(f1,f2)
        return w 


class Loss_Function_new_v2(object):  
    def __init__(self, delta, gama):  
        self.delta = delta
        self.gama = gama
        
    def calculate_loss(self, y, R, Rl):

        loss_per_sample_a = torch.exp(-2 * y * Rl)
        average_loss_a = torch.mean(loss_per_sample_a)

        loss_per_sample_b = torch.square(y * Rl - torch.abs(torch.tanh(R))) 
        average_loss_b = torch.mean(loss_per_sample_b)

        loss = self.delta*average_loss_a + self.gama*average_loss_b
        # print(average_loss_a, average_loss_b)
        return loss 


class SparseMax(object):  
    def __init__(self, beta, a, dim): 
        self.beta = beta 
        self.a = a  
        self.dim = dim
    def f(self, r):
        robust = torch.exp(self.beta * r)-self.a  
        return robust
    def forward(self, r):
        r = self.f(r)  
        r_sum = torch.sum(r,dim=self.dim,keepdim=True)  

        if torch.sum(r_sum==0):   
            r_sum[r_sum==0] = 1
        robust = torch.div(r,r_sum)

        return robust


class NormRobust(object): 
    def __init__(self, smax, scale):
        self.smax = smax
        self.scale = scale
    def forward(self, s, r, d):  #  w_norm, s, 1  
        eps = 1e-12
        r_w = s*r   #  
        # print(s.size(), r.size(), r_w.size())
        mx = torch.abs(torch.max(r_w,dim=d,keepdim=True)[0]) 
        r_re = self.scale*torch.div(r_w,(mx+eps)) 
        # r_re = r
        s_norm = self.smax.forward(r_re) 
        return s_norm


class Disjunction(object):  
    def __init__(self):
        pass
    def forward(self, X, w): 
        s = torch.clone(X)  
        w_sum = w.sum()
        if w_sum == 0:
            w_norm = w
        else:
            w_norm = w / w_sum  
        s_norm = self.normalize_robust.forward(w_norm, s, 1) 

        sw = torch.sum(torch.mul(s_norm,w_norm),dim=1) 

        if torch.any(sw==0):
            s_norm[sw==0,:] = 0.1  

        denominator = torch.mul(s_norm, w_norm)
        denominator = denominator
        denominator = torch.sum(denominator,dim=1,keepdim=True)   
        
        numerator = torch.mul(s_norm, w_norm)
        numerator = torch.mul(numerator, s)
        numerator = torch.sum(numerator,dim=1,keepdim=True)  

        denominator_old = torch.clone(denominator) 
        denominator[(denominator_old==0)] = 1  

        robust = numerator/denominator  

        if torch.sum(denominator_old == 0):
            if torch.all(denominator_old==0): # if all denominator equal zero
                robust[(denominator_old==0)] = -1
            else:
                robust[(denominator_old==0)] = torch.min(robust[(denominator_old!=0)])

        return robust

    def init_sparsemax(self, beta, a, scale, dim):
        self.smax = SparseMax(beta, a, dim)  
        self.normalize_robust = NormRobust(self.smax, scale)


class Conjunction(object): 
    def __init__(self):
        pass
    
    def forward(self, X, w):  
        s = torch.clone(-X) 
        w_sum = w.sum()
        if w_sum == 0:
            w_norm = w
        else:
            w_norm = w / w_sum 
        s_norm = self.normalize_robust.forward(w_norm, s, 1)
        sw = torch.sum(torch.mul(s_norm,w_norm),dim=1)

        if torch.any(sw==0):
            s_norm[sw==0,:] = 0.1

        denominator = torch.mul(s_norm, w_norm)
        denominator = torch.sum(denominator,dim=1,keepdim=True)
        
        numerator = torch.mul(s_norm, w_norm)
        numerator = torch.mul(numerator, s)
        numerator = torch.sum(numerator,dim=1,keepdim=True)
        denominator_old = torch.clone(denominator)

        numerator_old = torch.clone(numerator)
        
        denominator[(denominator_old==0)] = 1
        robust = -numerator/denominator
        if torch.sum(denominator_old == 0): # there exists zero denominator
            if torch.all(denominator_old==0): # if all denominator equal zero
                robust[(denominator_old==0)] = -1
            else:
                robust[(denominator_old==0)] = torch.min(robust[(denominator_old!=0)])
        return robust

    def init_sparsemax(self, beta, a, scale, dim):
        self.smax = SparseMax(beta, a, dim)
        self.normalize_robust = NormRobust(self.smax, scale)


class Eventually_and_Always(object):
    def __init__(self, scale, beta, NormRobust_New):
        self.normalize_robust = NormRobust_New(scale, beta, dim=2)

    def robustness(self, P_robust_all, W1s, mask, normalize_robusts):
        r_1 = P_robust_all 

        r_1 = torch.mul(r_1, mask) #
        
        row_means = W1s.sum(dim=1)

        w_norm = torch.zeros_like(W1s)

        for i in range(len(W1s)):
            if row_means[i]==0:
                w_norm[i,:] = W1s[i,:]
            else:
                w_norm[i,:] = W1s[i,:]/row_means[i]

        q_robust = normalize_robusts.forward(w_norm, r_1, 2)   

        sw = torch.sum(torch.mul(q_robust,w_norm), dim=2) 
        if torch.any(sw==0):
            q_robust[sw==0,:] = 0.1
        
        q_w_norm = torch.mul(q_robust, w_norm)  

        denominator = torch.sum(q_w_norm,dim=2,keepdim=True)  
        denominator_old = torch.clone(denominator)
        denominator[(denominator_old==0)] = 1
        numerator = torch.mul(q_w_norm, r_1)
        numerator = torch.sum(numerator,dim=2,keepdim=True)  
        
        robust = numerator/denominator 
        robust = torch.mul(robust, mask)
        
        if torch.sum(denominator_old == 0): # there exists zero denominator
            if torch.all(denominator_old==0): # if all denominator equal zero
                robust[(denominator_old==0)] = -1
            else:
                robust[(denominator_old==0)] = torch.min(robust[(denominator_old!=0)])

        return robust
    
    def calculate(self, x, a, b, W1s, mask):

        Ar = a.repeat(x.size(0),1,1)    
        # print(a.size(), Ar.size(), x.size(), b.size())  # torch.Size([1, 6]) torch.Size([10, 1, 6]) torch.Size([10, 6, 100]) torch.Size([1])
        P_robust_all = torch.matmul(Ar,x) - b  # [10,n,61]  12a

        robust = self.robustness(P_robust_all, W1s, mask, self.normalize_robust) 

        return robust

class NormRobust_New(object): 
    def __init__(self, scale, beta, dim):
        self.scale = scale
        self.beta = beta 
        self.dim = dim

    def forward(self, w_norm, r, d=2):  #  w_norm, r_1, 1
        eps = 1e-12
        r_w = w_norm*r  

        mx = torch.abs(torch.max(r_w, dim=d, keepdim=True)[0]) 
        r_re = self.scale*torch.div(r_w,(mx+eps)) 

        r_re_b = torch.exp(self.beta * r_re)    
        r_sum = torch.sum(r_re_b, dim=self.dim, keepdim=True)  
        q_robust = torch.div(r_re_b, r_sum) 

        return q_robust
    
