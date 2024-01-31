
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
    
class STEstimator(torch.autograd.Function): # 将input的值限制在[0, 1]之间，并映射为 1 和 0
    @staticmethod
    def forward(ctx, g):
        # g -> gs
        g_clip = torch.clamp(g, min=0, max = 1)  # clamp 将input的值限制在[min, max]之间  
        gs = g_clip.clone()  # clone()函数可以返回一个完全相同的tensor,新的tensor开辟新的内存
        gs[gs>=0.5] = 1
        gs[gs<0.5] = 0
        return gs
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clone(grad_output)
        return grad_input


class Clip(torch.autograd.Function):  # 将input的值映射为 1 和 -1
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


class RMinTimeWeight(object):  # ReLU 函数实现的时间窗
    def __init__(self, t1, t2):  # tau 被设置为1    t1 [8, 1]
        self.t1 = t1
        self.t2 = t2
        self.tau = torch.tensor(1, requires_grad=False)
        self.relu = nn.ReLU()
    def get_weight(self,w):
        f1 = (self.relu(w-self.t1+self.tau)-self.relu(w-self.t1))/self.tau
        f2 = (self.relu(-w+self.t2+self.tau)-self.relu(-w+self.t2))/self.tau
        w = torch.min(f1,f2)
        return w 


class Loss_Function_new_v2(object):  # sigmoid 函数实现的时间窗
    def __init__(self, delta, gama):  # tau 被设置为1    t1 [8, 1]
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


class SparseMax(object):  # Sparse softmax 函数 用于对max函数的近似，可微
    def __init__(self, beta, a, dim):  #  beta：控制SparseMax函数的平滑程度。 a：用于稀疏性的超参数。  dim：指定操作应用的维度。
        self.beta = beta 
        self.a = a  #0
        self.dim = dim
    def f(self, r):
        robust = torch.exp(self.beta * r)-self.a   # 减去了self.a 没什么用，论文里没说，设置的也是0 
        return robust
    def forward(self, r):
        r = self.f(r)  
        r_sum = torch.sum(r,dim=self.dim,keepdim=True)  

        if torch.sum(r_sum==0):   # 接着，检查是否存在r_sum中的所有元素都等于0，如果是，则将r_sum中的所有零值替换为1，以避免除零错误。
            r_sum[r_sum==0] = 1
        robust = torch.div(r,r_sum) #  计算SparseMax函数的输出，将robust除以r_sum，得到归一化后的。

        return robust


class NormRobust(object): # 鲁棒度归一化
    def __init__(self, smax, scale):
        self.smax = smax
        self.scale = scale
    def forward(self, s, r, d):  #  w_norm, s, 1  
        eps = 1e-12
        r_w = s*r   #  
        # print(s.size(), r.size(), r_w.size())
        mx = torch.abs(torch.max(r_w,dim=d,keepdim=True)[0])  # 计算 mx，它是在维度 d=1 上取 r_w 的绝对值的最大值
        r_re = self.scale*torch.div(r_w,(mx+eps)) #rescale r   这里计算对应得到r//  公式 12c
        # r_re = r
        s_norm = self.smax.forward(r_re) # weight of r_re  # 公式12d
        return s_norm


class Disjunction(object):  # 计算Disjunction公式的鲁棒度的
    def __init__(self):
        pass
    def forward(self, X, w): # OR, EVENTUALLY
        s = torch.clone(X)  #  创建一个tensor与源tensor有相同的shape，dtype和device，不共享内存地址，但新tensor的梯度会叠加在源tensor上
        w_sum = w.sum()
        if w_sum == 0:
            w_norm = w
        else:
            w_norm = w / w_sum  # 如果w_sum等于0，将w_norm设置为等于w，否则将w_norm计算为w除以w_sum，这是为了将权重w进行归一化
        s_norm = self.normalize_robust.forward(w_norm, s, 1)  # NormRobust中的forward方法 

        sw = torch.sum(torch.mul(s_norm,w_norm),dim=1) # 它是w_norm与s_norm的点积（逐元素相乘）然后在维度1上求和

        if torch.any(sw==0):
            s_norm[sw==0,:] = 0.1  # 检查是否存在sw中的任何元素等于0，如果是，则将s_norm中对应的行的所有元素设置为0.1

        denominator = torch.mul(s_norm, w_norm)
        denominator = denominator
        denominator = torch.sum(denominator,dim=1,keepdim=True)   # 计算denominator，它是w_norm与s_norm的逐元素相乘，然后在维度1上求和，得到一个列向量。
        
        numerator = torch.mul(s_norm, w_norm)
        numerator = torch.mul(numerator, s)
        numerator = torch.sum(numerator,dim=1,keepdim=True)  # 它是w_norm、s_norm和s的逐元素相乘，然后在维度1上求和，得到一个列向量。

        denominator_old = torch.clone(denominator) 
        denominator[(denominator_old==0)] = 1  # 然后将denominator中等于0的元素设置为1，以避免除零错误。

        robust = numerator/denominator  # 计算robust，它是numerator除以denominator，得到的是Disjunction公式的鲁棒度。

        if torch.sum(denominator_old == 0): # there exists zero denominator  最后，检查是否有零分母存在，如果有，则根据情况将robust中相应的元素设置为-1或最小非零值
            if torch.all(denominator_old==0): # if all denominator equal zero
                robust[(denominator_old==0)] = -1
            else:
                robust[(denominator_old==0)] = torch.min(robust[(denominator_old!=0)])

        return robust

    def init_sparsemax(self, beta, a, scale, dim):
        self.smax = SparseMax(beta, a, dim)  
        self.normalize_robust = NormRobust(self.smax, scale)


class Conjunction(object): # AND, ALWAYS
    def __init__(self):
        pass
    
    def forward(self, X, w): # OR, EVENTUALLY    # [400 , 8]  W [1, 8] 
        s = torch.clone(-X)  # [400 , 8]
        w_sum = w.sum()
        if w_sum == 0:
            w_norm = w
        else:
            w_norm = w / w_sum  # [1, 8]
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
        # evnt  r_1 = torch.clone(r)  alway r_1 = torch.clone(-r)

        # 计算每一行的均值  这段代码可以再改一下提升效率
        row_means = W1s.sum(dim=1)

        w_norm = torch.zeros_like(W1s)
        # 使用逻辑判断来处理除 0 的情况
        for i in range(len(W1s)):
            if row_means[i]==0:
                w_norm[i,:] = W1s[i,:]
            else:
                w_norm[i,:] = W1s[i,:]/row_means[i]

        q_robust = normalize_robusts.forward(w_norm, r_1, 2)  # q_robust  12d的计算结果

        sw = torch.sum(torch.mul(q_robust,w_norm), dim=2) # 12e的分母，不应该等于0
        if torch.any(sw==0):
            q_robust[sw==0,:] = 0.1
        # 
        q_w_norm = torch.mul(q_robust, w_norm)  

        denominator = torch.sum(q_w_norm,dim=2,keepdim=True) # 
        denominator_old = torch.clone(denominator)
        denominator[(denominator_old==0)] = 1
        numerator = torch.mul(q_w_norm, r_1)
        numerator = torch.sum(numerator,dim=2,keepdim=True)  # [10, n, 1]
        
        robust = numerator/denominator # 这一部分对应公式12e
        robust = torch.mul(robust, mask)
        
        if torch.sum(denominator_old == 0): # there exists zero denominator
            if torch.all(denominator_old==0): # if all denominator equal zero
                robust[(denominator_old==0)] = -1
            else:
                robust[(denominator_old==0)] = torch.min(robust[(denominator_old!=0)])

        return robust
    
    def calculate(self, x, a, b, W1s, mask):

        Ar = a.repeat(x.size(0),1,1) #   
        # print(a.size(), Ar.size(), x.size(), b.size())  # torch.Size([1, 6]) torch.Size([10, 1, 6]) torch.Size([10, 6, 100]) torch.Size([1])
        P_robust_all = torch.matmul(Ar,x) - b  # [10,n,61]  12a

        robust = self.robustness(P_robust_all, W1s, mask, self.normalize_robust) 

        return robust

class NormRobust_New(object): # 鲁棒度归一化
    def __init__(self, scale, beta, dim):
        self.scale = scale
        self.beta = beta 
        self.dim = dim

    def forward(self, w_norm, r, d=2):  #  w_norm, r_1, 1
        eps = 1e-12
        r_w = w_norm*r   #  12a

        mx = torch.abs(torch.max(r_w, dim=d, keepdim=True)[0])  #  d=1   12b
        r_re = self.scale*torch.div(r_w,(mx+eps)) #rescale   公式12c

        r_re_b = torch.exp(self.beta * r_re)    # 公式12d  r_re_b存在为0的情况
        r_sum = torch.sum(r_re_b, dim=self.dim, keepdim=True)  # 公式12d
        q_robust = torch.div(r_re_b, r_sum) #  

        return q_robust
    
