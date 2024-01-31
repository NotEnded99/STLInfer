
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

def convert_to_binary_labels(label, se_label): # 二分类，对标签进行划分 
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
    for i in range(data.shape[0]):  # 遍历每个样本
        for j in range(data.shape[1]):  # 遍历每个通道
            # 对数据进行归一化
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
            if tf == True:    # 这里记录第二个 j-1  需要是   w[j] = false 
                t12.append(j-1)
            else:
                t12.append(j)   # 这里记录第一个 j  需要是  w[j] = true 
            tf = not tf
            if len(t12) == 2:
                break
    if tf == True:
        t12.append(l-1)  
    tc1 = t12[0]
    tc2 = t12[1]

    return tc1, tc2

def extract_formula_new(x, y, a, b, Spatial, W1s, Wcs, Wds):  # 将对分类效果不显著的，权值设置为0 
    # _,_,acc_val = validation_accuracy(x,y,a,b,Formula1,conjunc,disjunc,clip,W1s,Wcs,Wds)  # 先计算得到 原始的 acc_val
    acc_val = STL_accuracy_new(x, y, a, b, Spatial, W1s, Wcs, Wds)
    f_num = W1s.shape[0]
    f_dis = Wds.shape[0]
    Wcs = Wcs.detach()
    Wds = Wds.detach()  # 将Wcs和Wds从计算图中分离，这可能是为了防止梯度回传对它们的影响

    for i in range(f_dis):
        if Wds[i]==0:  # 如果Wds[i]的值为0，表示该特征已经被标记为不重要，直接跳过。
            continue
        Wds_new = torch.clone(Wds) # 否则，将Wds的副本复制到Wds_new，然后将Wds_new[i]的值设置为0，表示将该特征的权重置为0。
        Wds_new[i] = 0  
        # _,_,acc_new = validation_accuracy(x,y,a,b,Formula1,conjunc,disjunc,clip,W1s,Wcs,Wds_new) # 再次调用validation_accuracy函数，评估模型在新的特征权重下的准确性，将结果存储在acc_new中。
        acc_new = STL_accuracy_new(x, y, a, b, Spatial, W1s, Wcs, Wds_new)
        # print(acc_new, acc_val)
        # print(round(acc_val.item(), 4))
        if round(acc_new.item(), 3) >= round(acc_val.item(), 3):   # 如果新的准确性与之前的准确性相同（四舍五入到小数点后两位），则将Wds[i]设置为0，表示该特征不重要。 如果某个特征的Wds值已经被置为0，就不再考虑它。
            Wds[i] = 0
        else: 
            for j in range(f_num):   # 对于每一个Wcs的元素，进行类似的操作，依次将元素设置为0，然后重新评估模型的准确性，如果准确性没有变化，就将该元素设置为0, 如果变化了 就保持原来的值
                if Wcs[i,j]==0:
                    continue
                Wcs_new = torch.clone(Wcs)
                Wcs_new[i,j] = 0
                # _,_,acc_new = validation_accuracy(x,y,a,b,Formula1,conjunc,disjunc,clip,W1s,Wcs_new,Wds)
                if sum(Wcs_new[i])>0:  # 避免Wcs_new全为0
                    acc_new = STL_accuracy_new(x, y, a, b, Spatial, W1s, Wcs_new, Wds)
                    if round(acc_new.item(), 3) >= round(acc_val.item(), 3):
                        Wcs[i,j] = 0   
    return Wcs, Wds

def extract_formula(x, y, a, b, Formula1, conjunc, disjunc, clip, W1s, Wcs, Wds):  # 将对分类效果不显著的，权值设置为0 
    _,_,acc_val = validation_accuracy(x,y,a,b,Formula1,conjunc,disjunc,clip,W1s,Wcs,Wds)  # 先计算得到 原始的 acc_val
    f_num = W1s.shape[0]
    f_dis = Wds.shape[0]
    Wcs = Wcs.detach()
    Wds = Wds.detach()  # 将Wcs和Wds从计算图中分离，这可能是为了防止梯度回传对它们的影响

    for i in range(f_dis):
        if Wds[i]==0:  # 如果Wds[i]的值为0，表示该特征已经被标记为不重要，直接跳过。
            continue
        Wds_new = torch.clone(Wds) # 否则，将Wds的副本复制到Wds_new，然后将Wds_new[i]的值设置为0，表示将该特征的权重置为0。
        Wds_new[i] = 0  
        _,_,acc_new = validation_accuracy(x,y,a,b,Formula1,conjunc,disjunc,clip,W1s,Wcs,Wds_new) # 再次调用validation_accuracy函数，评估模型在新的特征权重下的准确性，将结果存储在acc_new中。
        if round(acc_new.item(), 2) == round(acc_val.item(), 2):   # 如果新的准确性与之前的准确性相同（四舍五入到小数点后两位），则将Wds[i]设置为0，表示该特征不重要。 如果某个特征的Wds值已经被置为0，就不再考虑它。
            Wds[i] = 0
        else: 
            for j in range(f_num):   # 对于每一个Wcs的元素，进行类似的操作，依次将元素设置为0，然后重新评估模型的准确性，如果准确性没有变化，就将该元素设置为0, 如果变化了 就保持原来的值
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
    for k, formula in enumerate(Formula1):   # 调用formula.robustness_trace方法，计算输入数据x在特征权重W1s[k,:]下的鲁棒性轨迹，但只需要返回t1-t2时间段的鲁棒度。 将计算得到的鲁棒性值存储在r1o的第k列中
        xo1 = formula.robustness_trace(x,a[k],b[k],W1s[k,:],nsample,need_trace=False)
        r1o[:,k,:] = xo1[:,0]
    r2i = torch.squeeze(r1o,dim=2)  # 压缩r1o的最后一个维度，得到r2i，其形状为(nsample, f_num)。
    
    r2o = torch.empty((nsample,f_conj), dtype=torch.float)

    for k in range(f_conj):
        xo2 = conjunc.forward(r2i,Wcs[k,:])
        r2o[:,k] = xo2[:,0]

    R = disjunc.forward(r2o,Wds)
    Rl = clip(R)  # 对鲁棒性值R进行截断操作，以确保其值为-1 或 1 

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
            xo1 = xo1[:,:,W1s[k,:]==1]     # W1s是由t1和t2得到的指示函数值

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

    