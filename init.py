import numpy as np
import torch
import os
import random 

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

def inverse_normalize_value_b(min_values, max_values, a, b, new_min=-1, new_max=1):
    min_values = torch.tensor(min_values, requires_grad=False)
    max_values = torch.tensor(max_values, requires_grad=False)

    temp = torch.cat((min_values, max_values), dim=0).view(2, -1)
    # print(temp, temp.size())
    min_max_value = torch.matmul(a, torch.transpose(temp, 0, 1)) # calculate   
    min_max_value = min_max_value.squeeze()
    # print(min_max_value.size())
    # print(min_max_value)
    value_b = torch.zeros_like(b)

    for i in range(len(b)):
         value_b[i] = ((b[i] - new_min) / (new_max - new_min)) * (min_max_value[i, 1] - min_max_value[i, 0]) + min_max_value[i, 0]
    return value_b


