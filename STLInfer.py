
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pickle
import time
import math
from STL_NN import *
from utils import *

torch.set_default_dtype(torch.float64)

def STLInfer_StageOne(train_data, train_label, val_data, val_label, n_dataset, i_class, k_run, args):
    nsample = train_data.shape[0]
    dim = train_data.shape[1]     
    length = train_data.shape[-1] 
    
    # initialization functions
    a_round = Transform_Param_A.apply
    STE= STEstimator.apply  
    clip = Clip.apply 
    
    f_num = 2*4*dim  # number of predicates 
    fn = int(f_num/4) 

    initial_a_value = Initialize_Parameter_A(dim)
    initial_a_value = torch.tensor(initial_a_value, dtype=torch.float64, requires_grad=True)
    a = torch.cat((initial_a_value, initial_a_value), axis=0) 
    a = torch.tensor(a, dtype=torch.float64, requires_grad=True)

    t1 = np.floor(np.random.rand(f_num,1)*(length-1)) 
    t2 = np.floor(np.random.rand(f_num,1)*(length-1))
    for k in range(len(t1)):
        if t1[k]>(t2[k]-1):
            t1[k] = t2[k]-1
    t1 = torch.tensor(t1, requires_grad=True)
    t2 = torch.tensor(t2, requires_grad=True)
    
    Wc = torch.ones((args.f_conj, f_num), requires_grad=True)
    Wd = torch.ones(args.f_conj, requires_grad=False) 

    b = torch.rand((f_num,1))*2-1 
    b.requires_grad=True

    W = RMinTimeWeight(t1,t2)  
    W1 = torch.tensor(range(length), requires_grad=False) 

    temporal_operators = []
    mask = []

    beta = 1
    scale = calculate_scale_value(length=length)
    eventually_and_always  = Eventually_and_Always(scale, beta, NormRobust_New)

    for k in range(2):
        for i in range(fn):
            mask.append(1.0)
            temporal_operators.append('F') 
        for i in range(fn):
            mask.append(-1.0)
            temporal_operators.append('G')

    mask = torch.tensor(mask)
    mask = mask.unsqueeze(1)
    
    scale = calculate_scale_value(length=f_num) # number of formulations 
    am = 0
    conjunc = Conjunction()  
    conjunc.init_sparsemax(beta, am, scale,1)

    scale = calculate_scale_value(length=args.f_conj)
    disjunc = Disjunction()   
    disjunc.init_sparsemax(beta, am, scale, 1)

    optimizer1 = torch.optim.Adam([a], lr=args.learning_rate_1)
    optimizer2 = torch.optim.Adam([Wc], lr=args.learning_rate_1)    
    optimizer3 = torch.optim.Adam([b], lr=args.learning_rate_1)   
    optimizer4 = torch.optim.Adam([t1,t2], lr=length*args.learning_rate_2) 

    loss_function = Loss_Function_new_v2(delta=1, gama=0.25)

    batch_size = args.batch_size

    acc = 0
    acc_best = 0
    extract_acc_best = 0


    for epoch in range(1, args.n_iters_1):
        a_epoch_loss = 0 

        rand_num = rd.sample(range(0,nsample),batch_size) 
        x = train_data[rand_num,:,:]  
        y = train_label[rand_num]

        aa = a_round(a) 
        with torch.no_grad():
            aa[0:fn*2] = initial_a_value

        Wt = W.get_weight(W1) 
        W1s = STE(Wt)
        Wcs = STE(Wc)  

        r1o = torch.empty((batch_size,f_num,1)) # f_num = 8
        r1o = eventually_and_always.calculate(x, aa, b, W1s, mask)
        r2i = torch.squeeze(r1o)  # [10, 8]
        r2o = torch.empty((batch_size, args.f_conj)) # [10, 1]

        for k in range(args.f_conj):
            if sum(Wcs[k,:])==0: 
                Wd[k] = 0  
            else:
                Wd[k] = 1  
            xo2 = conjunc.forward(r2i,Wcs[k,:])
            r2o[:,k] = xo2[:,0]  # r2o [10, 1] 

        Wd = torch.sum(Wcs,1)
        Wds = clip(Wd)
        R = disjunc.forward(r2o,Wds)
        Rl = clip(R)
        Rl = Rl[:,0]

        l = loss_function.calculate_loss(y, R, Rl)
        a_epoch_loss = l.detach().numpy()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        l.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        with torch.no_grad():
            Wc[Wc<=0] = 0
            Wc[Wc>=1] = 1

        with torch.no_grad():
            t1[t1<0] = 0
            t2[t2<1] = 1 
            t1[t1>length-1] = length-2   
            t2[t2>length-1] = length-1
            for k, t in enumerate(t1):
                if t>(t2[k]-1):
                    t1[k] = t2[k]-1
        with torch.no_grad():
            for k in range(2*fn):
                if b[k]<-1:
                    b[k]=-1
                if b[k]>1:
                    b[k]=1

        if epoch % 10 ==0:  
            val_nsample = val_data.shape[0] 
            r1o = torch.empty((val_nsample,f_num,1))
            r1o = eventually_and_always.calculate(val_data, aa, b, W1s, mask)
            r2i = torch.squeeze(r1o)
            r2o = torch.empty((val_nsample, args.f_conj))
            for k in range(args.f_conj):
                xo2 = conjunc.forward(r2i, Wcs[k,:])
                r2o[:,k] = xo2[:,0]
            R = disjunc.forward(r2o, Wds)
            Rl = clip(R)
            acc_val = sum(val_label==Rl[:,0])/(val_nsample)

            # calculate the true accuracy 
            acc_stl = STL_accuracy_new(val_data, val_label, aa, b, temporal_operators, W1s, Wcs, Wds)
            train_acc_stl = STL_accuracy_new(train_data, train_label, aa, b, temporal_operators, W1s, Wcs, Wds)
            acc = acc_stl
            
            print('epoch_num = {epoch}, loss = {l}, accracy_train = {accracy_train}, accuracy_val = {acc_val}, accuracy_stl = {acc_stl}'.format(epoch=epoch, 
                                     l=a_epoch_loss, accracy_train=train_acc_stl, acc_val=acc_val, acc_stl=acc_stl))
            if acc>acc_best:
                        acc_best = acc   
                        if extract_acc_best==0:
                            extract_acc_best = acc        

                        Wcss, Wdss =  extract_formula_new(train_data, train_label, aa, b, temporal_operators, W1s, Wcs, Wds)
                        extract_formula_accuracy = STL_accuracy_new(val_data, val_label, aa, b, temporal_operators, W1s, Wcss, Wdss)
                        print(" extract_formula_accuracy = ", extract_formula_accuracy.numpy())
                        
                        file_path = args.model_dir_1
                        if  not os.path.exists(file_path):
                            os.makedirs(file_path)

                        if extract_formula_accuracy >= extract_acc_best:
                            extract_acc_best = extract_formula_accuracy
                            f = open(args.model_dir_1+'\W_best_'+n_dataset+'_{i}_{k}.pkl'.format(i=i_class, k=k_run), 'wb')
                            pickle.dump([W1s, Wcss, Wdss, aa, b, t1, t2, temporal_operators, extract_acc_best], f)
                            f.close()
                        elif acc_best >= extract_acc_best:
                            extract_acc_best = acc
                            Wcss = Wcs.detach()
                            Wdss = Wds.detach()
                            f = open(args.model_dir_1+'\W_best_'+n_dataset+'_{i}_{k}.pkl'.format(i=i_class, k=k_run), 'wb')
                            pickle.dump([W1s, Wcss, Wdss, aa, b, t1, t2, temporal_operators, extract_acc_best], f)
                            f.close()   

    final_accuracy  = extract_acc_best
    return  final_accuracy

def STLInfer_StageTwo(train_data, train_label, val_data, val_label, n_dataset, i_class, k_run, args):
    print( 'the stage two is started!' )
    with open(args.model_dir_1+'\W_best_'+n_dataset+'_{i}_{k}.pkl'.format(i=i_class, k=k_run), 'rb') as f:
        W1s_old, Wcss_old, Wdss_old, a_old, b_old, t1_old, t2_old, Spatial_old, model_old_acc = pickle.load(f)
    old_acc = model_old_acc.item()

    if old_acc==1.00:
        before_loaca_search_accuracy=1.0
        final_accuracy=1.0
        best_training_time = 0.0
        f = open(args.model_dir_2+'\W_best_'+n_dataset+'_{i}_{k}.pkl'.format(i=i_class, k=k_run), 'wb')
        pickle.dump([W1s_old, Wcss_old, Wdss_old, a_old, b_old, t1_old, t2_old, Spatial_old, model_old_acc], f)
        f.close()
        return before_loaca_search_accuracy, final_accuracy, best_training_time
    
    Wc = Wcss_old.detach().numpy()
    indices_to_keep = np.where(Wc[0] == 1)[0]
    f_num = len(indices_to_keep)

    Spatial = [Spatial_old[i] for i in indices_to_keep]
    indices_to_keep = torch.tensor(indices_to_keep)

    Wcs = Wcss_old[:,indices_to_keep].detach()
    Wd = Wdss_old
    aa = a_old[indices_to_keep, :].detach()
    b = b_old[indices_to_keep, :].detach()
    t1 = t1_old[indices_to_keep, :].detach()
    t2 = t2_old[indices_to_keep, :].detach()

    aa.requires_grad=False
    Wcs.requires_grad=False

    b.requires_grad=True
    t1.requires_grad=True
    t2.requires_grad=True

    nsample = train_data.shape[0]
    length = train_data.shape[-1] 
    
    # initialization
    STE= STEstimator.apply  
    clip = Clip.apply 

    f_conj = 1 

    tl2 = length-1

    at = torch.tensor(1, requires_grad=False)  
    W1 = torch.tensor(range(length), requires_grad=False)

    mask = []

    beta = 1
    am = 0
    scale = calculate_scale_value(length=length) 

    eventually_and_always  = Eventually_and_Always(scale, beta, NormRobust_New)

    for i in range(len(Spatial)):
        if Spatial[i] == 'F':
            mask.append(1.0)
        elif Spatial[i] == 'G':
            mask.append(-1.0)
    
    mask = torch.tensor(mask)
    mask = mask.unsqueeze(1)

    am = 0
    scale = calculate_scale_value(length=f_num)

    conjunc = Conjunction()  
    conjunc.init_sparsemax(beta,am,scale,1)

    scale = calculate_scale_value(length=f_conj) 

    disjunc = Disjunction()   
    disjunc.init_sparsemax(beta,am,scale,1)

    optimizer1 = torch.optim.Adam([b], lr=0.1)  
    optimizer2 = torch.optim.Adam([t1,t2], lr=length*0.01)
                                  
    loss_function = Loss_Function_new_v2(delta=1, gama=0.25)

    batch_size = 10
    acc = 0
    acc_best = 0
    extract_acc_best = 0

    for epoch in range(1, args.n_iters_2):
        a_epoch_loss = 0 
        rand_num = rd.sample(range(0,nsample),batch_size) 

        x = train_data[rand_num,:,:]  
        y = train_label[rand_num]

        f1 = torch.relu(W1-t1+at)-torch.relu(W1-t1)
        f2 = torch.relu(-W1+t2+at)-torch.relu(-W1+t2)
        Wt = torch.min(f1,f2)
        W1s = STE(Wt)
        
        r1o = torch.empty((batch_size,f_num,1))

        r1o = eventually_and_always.calculate(x, aa, b, W1s, mask)

        r2i = torch.squeeze(r1o) 
        r2o = torch.empty((batch_size, f_conj)) 

        for k in range(f_conj):
            if sum(Wcs[k,:])==0: 
                Wd[k] = 0  
            else:
                Wd[k] = 1  
            if  r2i.dim() == 1:
                r2i = r2i.unsqueeze(1) 
            xo2 = conjunc.forward(r2i, Wcs[k,:])
            r2o[:,k] = xo2[:,0] 

        Wd = torch.sum(Wcs,1)
        Wds = clip(Wd)
        R = disjunc.forward(r2o,Wds)
        Rl = clip(R)
        Rl = Rl[:,0]

        l = loss_function.calculate_loss(y, R, Rl)
        l.requires_grad_(True)

        a_epoch_loss = l.detach().numpy()

        bb = b.clone().detach()
        tt1 = t1.clone().detach()
        tt2 = t2.clone().detach()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        l.backward()
        optimizer1.step()
        optimizer2.step()

        if f_num>1:
            num_ori = math.ceil(f_num*0.5)
            random_indices = torch.randperm(f_num)[: num_ori]
            with torch.no_grad():
                b[random_indices] = bb[random_indices] 
                b.requires_grad = True
                t1[random_indices] = tt1[random_indices] 
                t1.requires_grad = True
                t2[random_indices] = tt2[random_indices] 
                t2.requires_grad = True

        with torch.no_grad():
            t1[t1<0] = 0
            t2[t2<0] = 1 
            t1[t1>tl2] = tl2-1  
            t2[t2>tl2] = tl2
            for k, t in enumerate(t1):
                if t>(t2[k]-1):
                    t1[k] = t2[k]-1
            
        if epoch % 5 ==0:  
            val_nsample = val_data.shape[0] 

            r1o = torch.empty((val_nsample,f_num,1))
            r1o = eventually_and_always.calculate(val_data, aa, b, W1s, mask)
            r2i = torch.squeeze(r1o)
            r2o = torch.empty((val_nsample,f_conj))
            for k in range(f_conj):
                if r2i.dim() == 1:
                    r2i = r2i.unsqueeze(1)  
                xo2 = conjunc.forward(r2i,Wcs[k,:])
                r2o[:,k] = xo2[:,0]
            R = disjunc.forward(r2o,Wds)
            Rl = clip(R)
            acc_val = sum(val_label==Rl[:,0])/(val_nsample)

            # calculate the true accuracy 
            acc_stl = STL_accuracy_new(val_data, val_label, aa, b, Spatial, W1s, Wcs, Wds)
            train_acc_stl = STL_accuracy_new(train_data, train_label, aa, b, Spatial, W1s, Wcs, Wds)
            acc = acc_stl
            
            print('epoch_num = {epoch}, loss = {l}, accracy_train = {accracy_train}, accuracy_val = {acc_val}, accuracy_stl = {acc_stl}'.format(epoch=epoch,l=a_epoch_loss, 
                                                                accracy_train=train_acc_stl, acc_val=acc_val, acc_stl=acc_stl))
            if acc>acc_best:
                        acc_best = acc   
                        if extract_acc_best==0:
                            extract_acc_best = acc        
                        Wcss, Wdss =  extract_formula_new(train_data, train_label, aa, b, Spatial, W1s, Wcs, Wds)
                        extract_formula_accuracy = STL_accuracy_new(val_data, val_label, aa, b, Spatial, W1s, Wcss, Wdss)
                        print(" extract_formula_accuracy = ", extract_formula_accuracy.numpy())
                        
                        file_path = args.model_dir_2
                        if  not os.path.exists(file_path):
                            os.makedirs(file_path)
                        
                        if extract_formula_accuracy >= extract_acc_best:
                            extract_acc_best = extract_formula_accuracy
                            f = open(args.model_dir_2+'\W_best_'+n_dataset+'_{i}_{k}.pkl'.format(i=i_class, k=k_run), 'wb')
                            pickle.dump([W1s, Wcss, Wdss, aa, b, t1, t2, Spatial, extract_acc_best], f)
                            f.close()
                        elif acc_best >= extract_acc_best:
                            extract_acc_best = acc
                            Wcss = Wcs.detach()
                            Wdss = Wds.detach()
                            f = open(args.model_dir_2+'\W_best_'+n_dataset+'_{i}_{k}.pkl'.format(i=i_class, k=k_run), 'wb')
                            pickle.dump([W1s, Wcss, Wdss, aa, b, t1, t2, Spatial, extract_acc_best], f)
                            f.close()  

    if extract_acc_best<old_acc:
        extract_acc_best = old_acc
        f = open(args.model_dir_2+'\W_best_'+n_dataset+'_{i}_{k}.pkl'.format(i=i_class, k=k_run), 'wb')
        pickle.dump([W1s_old, Wcss_old, Wdss_old, a_old, b_old, t1_old, t2_old, Spatial_old, model_old_acc], f)
        f.close()
    final_accuracy  = extract_acc_best

    return final_accuracy
