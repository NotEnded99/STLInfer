
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import time 
import copy
import os 

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # set CPU seed
    torch.cuda.manual_seed(seed) # set GPU seed 
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    np.random.seed(seed) # Numpy module.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def convert_to_binary_labels(label, se_label):
    outlabel = np.copy(label)
    outlabel[outlabel!=se_label]=-1
    outlabel[outlabel==se_label]=1
    return outlabel

def find_indices(arr, value1, value2):
    indices = np.where((arr == value1) | (arr == value2))
    return indices

def calculate_dimension_minmax(train_data, test_data):

    train_max_values = np.max(train_data, axis=2)
    train_min_values = np.min(train_data, axis=2)
    train_max_values1 = np.max(train_max_values, axis=0)
    train_min_values1 = np.min(train_min_values, axis=0)
    
    test_max_values = np.max(test_data, axis=2)
    test_min_values = np.min(test_data, axis=2)
    test_max_values1 = np.max(test_max_values, axis=0)
    test_min_values1 = np.min(test_min_values, axis=0)

    overall_max = np.max(np.vstack((train_max_values1, test_max_values1)), axis=0)
    overall_min = np.min(np.vstack((train_min_values1, test_min_values1)), axis=0)
    return overall_max, overall_min

def minmax_normalize_samples(data, min_vals, max_vals, new_min=-1, new_max=1):
    
    normalized_data = np.zeros_like(data)
    # normalize each sample
    for i in range(data.shape[0]):  
        for j in range(data.shape[1]):  
           
            normalized_data[i, j, :] = (data[i, j, :] - min_vals[j]) / (max_vals[j] - min_vals[j])*(new_max - new_min) + new_min
    return normalized_data

def calculate_scale_value(length):
    scale = 1
    while scale*np.exp(scale)<(length*np.exp(-1)):
        scale = scale + 0.1
    return round(scale, 3)

def get_t1_t2(w):  
    w = w.bool()
    l = w.shape[0]
    t = []
    t12 = []
    tf = False
    for j in range(l):
        if w[j] != tf:
            if tf == True:    
                t12.append(j-1)
            else:
                t12.append(j)   
            tf = not tf
            if len(t12) == 2:
                break
    if tf == True:
        t12.append(l-1)  
    tc1 = t12[0]
    tc2 = t12[1]

    return tc1, tc2

def extract_formula_new(x, y, a, b, Spatial, W1s, Wcs, Wds): 
    # _,_,acc_val = validation_accuracy(x,y,a,b,Formula1,conjunc,disjunc,clip,W1s,Wcs,Wds) 
    acc_val = STL_accuracy_new(x, y, a, b, Spatial, W1s, Wcs, Wds)
    f_num = W1s.shape[0]
    f_dis = Wds.shape[0]
    Wcs = Wcs.detach()
    Wds = Wds.detach() 

    for i in range(f_dis):
        if Wds[i]==0: 
            continue
        Wds_new = torch.clone(Wds) 
        Wds_new[i] = 0  
        # _,_,acc_new = validation_accuracy(x,y,a,b,Formula1,conjunc,disjunc,clip,W1s,Wcs,Wds_new) 
        acc_new = STL_accuracy_new(x, y, a, b, Spatial, W1s, Wcs, Wds_new)
        # print(acc_new, acc_val)
        # print(round(acc_val.item(), 4))
        if round(acc_new.item(), 3) >= round(acc_val.item(), 3):  
            Wds[i] = 0
        else: 
            for j in range(f_num):  
                if Wcs[i,j]==0:
                    continue
                Wcs_new = torch.clone(Wcs)
                Wcs_new[i,j] = 0
                # _,_,acc_new = validation_accuracy(x,y,a,b,Formula1,conjunc,disjunc,clip,W1s,Wcs_new,Wds)
                if sum(Wcs_new[i])>0:  
                    acc_new = STL_accuracy_new(x, y, a, b, Spatial, W1s, Wcs_new, Wds)
                    if round(acc_new.item(), 3) >= round(acc_val.item(), 3):
                        Wcs[i,j] = 0   
    return Wcs, Wds

def extract_formula(x, y, a, b, Formula1, conjunc, disjunc, clip, W1s, Wcs, Wds): 
    _,_,acc_val = validation_accuracy(x,y,a,b,Formula1,conjunc,disjunc,clip,W1s,Wcs,Wds) 
    f_num = W1s.shape[0]
    f_dis = Wds.shape[0]
    Wcs = Wcs.detach()
    Wds = Wds.detach()

    for i in range(f_dis):
        if Wds[i]==0: 
            continue
        Wds_new = torch.clone(Wds)
        Wds_new[i] = 0  
        _,_,acc_new = validation_accuracy(x,y,a,b,Formula1,conjunc,disjunc,clip,W1s,Wcs,Wds_new) 
        if round(acc_new.item(), 2) == round(acc_val.item(), 2):  
            Wds[i] = 0
        else: 
            for j in range(f_num):  
                if Wcs[i,j]==0:
                    continue
                Wcs_new = torch.clone(Wcs)
                Wcs_new[i,j] = 0
                _,_,acc_new = validation_accuracy(x,y,a,b,Formula1,conjunc,disjunc,clip,W1s,Wcs_new,Wds)
                if round(acc_new.item(), 2) == round(acc_val.item(), 2):
                    Wcs[i,j] = 0   
    return Wcs, Wds

def validation_accuracy(x,y,a,b,Formula1,conjunc,disjunc,clip,W1s,Wcs,Wds):
    nsample = x.shape[0]  # batch size 
    f_num = W1s.shape[0]
    f_conj = Wcs.shape[0]
    r1o = torch.empty((nsample,f_num,1), dtype=torch.float)
    for k, formula in enumerate(Formula1):   
        xo1 = formula.robustness_trace(x,a[k],b[k],W1s[k,:],nsample,need_trace=False)
        r1o[:,k,:] = xo1[:,0]
    r2i = torch.squeeze(r1o,dim=2)  
    
    r2o = torch.empty((nsample,f_conj), dtype=torch.float)

    for k in range(f_conj):
        xo2 = conjunc.forward(r2i,Wcs[k,:])
        r2o[:,k] = xo2[:,0]

    R = disjunc.forward(r2o,Wds)
    Rl = clip(R)  

    acc = sum(y==Rl[:,0])/(nsample)
    false_data = x[y!=Rl[:,0],:,:]
    false_label = y[y!=Rl[:,0]]
    return false_data, false_label, acc


def STL_accuracy_new(x, y, a, b, Spatial, W1s, Wcs, Wds):
    if sum(Wds)==0:
        acc = torch.tensor(0.0, requires_grad=False)
        return acc
    else:
        nsample = x.shape[0]
        f_num = W1s.shape[0]
        f_conj = Wcs.shape[0]
        r1o = torch.empty((nsample,f_num,1), dtype=torch.float)

        for k in range(len(b)):
            Ar = a[k].repeat(x.size(0), 1, 1) 
            xo1 = torch.matmul(Ar,x) - b[k]  
            
            # print("xo1", xo1)
            xo1 = xo1[:,:,W1s[k,:]==1]    

            if Spatial[k]=='F':
                # print(xo1)
                r1o[:,k,:] = torch.max(xo1,2)[0]
                
            elif Spatial[k]=='G':
                r1o[:,k,:] = torch.min(xo1,2)[0]   
        
        r2i = torch.squeeze(r1o,dim=2)
        r2o = torch.empty((nsample,f_conj), dtype=torch.float)

        for k in range(f_conj):
            xo2 = r2i[:,Wcs[k,:]==1]   
            r2o[:,k] = torch.min(xo2,1)[0]
        ro = r2o[:,Wds==1]  

        R = torch.max(ro,1)[0]

        R[R>0] = 1
        R[R<=0] = -1
        acc = sum(y==R)/(nsample)
        return acc

    
