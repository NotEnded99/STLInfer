
import torch
import numpy as np
from STLInfer import *  
from get_learned_formula import *
from utils import *
import argparse
import json
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=r'.\datasets', help='The directory of the dataset')
parser.add_argument('--model_dir_1', type=str, default=r'.\models_stage_one', help='The directory to save the model of stage one')
parser.add_argument('--model_dir_2', type=str, default=r'.\models_stage_two', help='The directory to save the model of stage two')
parser.add_argument('--result_dir', type=str, default=r'.\results', help='The directory to save the results')
parser.add_argument('--dataset_list', type=str, default=['Naval', 'BasicMotions', 'Epilepsy',  'Blink',  'ERing', 'FingerMovements', 
                    'SelfRegulationSCP1', 'SelfRegulationSCP2','UWaveGestureLibrary', 'NATOPS'], help='A list of all dataset names')
parser.add_argument('--dataset_list1', type=str, default=['Epilepsy'], help='A list of all dataset names')
parser.add_argument('--batch_size', type=int, default=20, help='The batch size')
parser.add_argument('--num_trial', type=int, default=5, help='The number of trials')
parser.add_argument('--n_iters_1', type=int, default=1000, help='The number of epochs in stage 1')
parser.add_argument('--n_iters_2', type=int, default=200, help='The number of epochs in stage 2')
parser.add_argument('--learning_rate_1', type=float, default=1e-1, help='The learning rate of a, b, wc')
parser.add_argument('--learning_rate_2', type=float, default=1e-2, help='The learning rate of t')
parser.add_argument('--f_conj', type=int, default=1, help='number of sub-formula for disconjunction')
args = parser.parse_args()

torch.set_default_dtype(torch.float64)

# modelname = 'model_v12_v4' # model_v12_v3_ori model_v12_v4 'model_v15_bp'， model_v17_pso

# work_dir = r'C:\Users\73175\Desktop\2023_paper_Learning_STL_via_NN\dataset\UCR_Processed'
# save_dir = r'C:\Users\73175\Desktop\2023_paper_Learning_STL_via_NN\models\Mine_model_v12_v4'
# work_dir = r'C:\Users\73175\Documents\工作文档\2023_paper_Learning_STL_via_NN\dataset\UCR_Processed'

# name_list = ['Naval', 'BasicMotions', 'Epilepsy',  'Blink',  'ERing', 'FingerMovements', 
#              'SelfRegulationSCP1', 'SelfRegulationSCP2','UWaveGestureLibrary', 'NATOPS']

# name_list = ['Epilepsy']

# number_of_trial = 5

all_dataset_time = []
all_dataset_time_mean = []

for index, name in enumerate(args.dataset_list1):
    print("start the {i} dataset search".format(i=name))
    train_data = np.load(args.data_dir+'\\'+name+'\\'+'train_data.npy')
    test_data = np.load(args.data_dir+'\\'+name+'\\'+'test_data.npy') 
    train_label = np.load(args.data_dir+'\\'+name+'\\'+'train_label.npy')
    test_label = np.load(args.data_dir+'\\'+name+'\\'+'test_label.npy')


    overall_max, overall_min = calculate_dimension_minmax(train_data, test_data)

    train_data = minmax_normalize_samples(train_data, overall_min, overall_max, new_min=-1, new_max=1)
    test_data = minmax_normalize_samples(test_data, overall_min, overall_max, new_min=-1, new_max=1)

    # trasfer narray into tensor
    train_data = torch.tensor(train_data, requires_grad=False)
    test_data = torch.tensor(test_data, requires_grad=False)

    results_stage_one = []
    results_stage_two = []
    one_dataset_formula = []
    one_dataset_time = []

    unique_elements = np.unique(train_label)

    for i in range(len(unique_elements)):
        if len(unique_elements)==2:
            if i==1:
                continue
        se_label = i
        _train_label = convert_to_binary_labels(train_label, se_label)
        _test_label = convert_to_binary_labels(test_label, se_label)

        # print(_test_label)
        # trasfer narray into tensor
        _train_label = torch.tensor(_train_label, requires_grad=False)
        _test_label = torch.tensor(_test_label, requires_grad=False)

        n_iters = 1000
        n_iters2 = 200 # local search

        one_class_results_stage_one = []
        one_class_results_stage_two = []

        one_class_5_formula = []

        for k in range(args.num_trial):
            print('the {i}-th class, the {k}-th run'.format(i=i, k=k))
            set_random_seed(k)
        
            start_time = time.time()
            stage_one_result = STLInfer_StageOne(train_data, _train_label, test_data, _test_label, name, i, k, args)
            end_time = time.time()
            execution_time_one = end_time - start_time
            one_class_results_stage_one.append(stage_one_result) 

        for k in range(args.num_trial):
            print('the {i}-th class, the {k}-th run'.format(i=i, k=k))
            set_random_seed(k)
    
            start_time = time.time()
            stage_two_result = STLInfer_StageTwo(train_data, _train_label, test_data, _test_label, name, i, k, args)
            end_time = time.time()
            execution_time_two = end_time - start_time
            one_dataset_time.append(execution_time_one+execution_time_two)
            one_class_results_stage_two.append(stage_two_result)

            _formula = new_get_formula_v1(args.model_dir_2, name, i, k)  
            one_class_5_formula.append(_formula)

        one_dataset_formula.append(one_class_5_formula)
        results_stage_one.append(one_class_results_stage_one)
        results_stage_two.append(one_class_results_stage_two)

    # the time of each dataset
    all_dataset_time.append(one_dataset_time)
    one_dataset_time = np.array(one_dataset_time)
    one_dataset_time_mean = np.mean(one_dataset_time)
    all_dataset_time_mean.append(one_dataset_time_mean)

    results_stage_one= np.array(results_stage_one)
    each_class_mean_stage_one = np.mean(results_stage_one, axis=1)  
    overall_mean_stage_one = np.mean(each_class_mean_stage_one)
    print('the stage one mean value of {i} dataset is: %.4f'.format(i=name) % overall_mean_stage_one)

    results_stage_two= np.array(results_stage_two)
    each_class_mean_stage_two= np.mean(results_stage_two, axis=1)  
    overall_mean_stage_two = np.mean(each_class_mean_stage_two)
    print('the stage two mean value of {i} dataset is: %.4f'.format(i=name) % overall_mean_stage_two)

    each_class_mean_stage_one = each_class_mean_stage_one.tolist()
    each_class_mean_stage_two = each_class_mean_stage_two.tolist()
    results_stage_one = results_stage_one.tolist()
    results_stage_two = results_stage_two.tolist()

    data = {'overall_mean_stage_one': overall_mean_stage_one, 'overall_mean_stage_two': overall_mean_stage_two,
             'each_class_mean_stage_one': each_class_mean_stage_one,  'each_class_stage_two': each_class_mean_stage_two,
               'results_stage_one':results_stage_one, 'results_stage_two':results_stage_two}

    formula = {"one_dataset_formula":one_dataset_formula}


    file_path = args.result_dir
    if  not os.path.exists(file_path):
        os.makedirs(file_path)

    file_path = args.result_dir+'\\data_{i}.json'.format(i=name)
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

    file_path = args.result_dir+'\\formula_{i}.json'.format(i=name)
    with open(file_path, 'w') as json_file:
        json.dump(formula, json_file)

time_json = {"all_dataset_time_mean":all_dataset_time_mean, "all_dataset_time":all_dataset_time}
file_path = args.result_dir+'\\computation_cost.json'
with open(file_path, 'w') as json_file:
    json.dump(time_json, json_file)

    


