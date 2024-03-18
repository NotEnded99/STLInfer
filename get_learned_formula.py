import numpy as np
import pickle




# def new_get_formula( Dataset_name):

#     with open('W_best_{i}.pkl'.format(i=Dataset_name), 'rb') as f:
#         W1s, Wcss, Wdss, a, b, t1, t2, Spatial, final_accuracy = pickle.load(f)

#     W1s = W1s.detach().numpy()
#     Wcss = Wcss.numpy()
#     Wdss = Wdss.numpy()
#     a = a.detach().numpy()
#     b = b.detach().numpy()
#     b = np.around(b, 3)
#     t1 = t1.detach().numpy()
#     t1 = np.around(t1, 1)
#     t2 = t2.detach().numpy()
#     t2 = np.around(t2, 1)
#     final_accuracy = np.around(final_accuracy, 4)
#     dim = np.shape(a)[-1]


#     Formula = r"\quad"
#     for i in range(len(Wcss)):
#         temp_formu = r"\quad"

#         for j in range(len(b)):
#             if Wcss[i, j] == 1:
#                 # if sum(a)!=0:
#                 variable = ""
#                 first_occurrence_index = np.where(W1s[j,:] == 1)[0][0] 
#                 last_occurrence_index = np.where(W1s[j,:] == 1)[0][-1] 
#                 for k in range(dim):
#                     if a[j, k]==1:
#                         variable += "+x_{k}".format(k=k)
#                     elif a[j, k]==-1:
#                         variable += "-x_{k}".format(k=k)
#                 variable1 = "({z})".format(z=variable)

#                 formu = r"{g}[{t1}, {t2}] {x} > {b}".format(g=Spatial[j], t1=first_occurrence_index, 
#                                                             t2=last_occurrence_index, x = variable1, b=b[j, i])
#                 if len(temp_formu)<8:
#                     temp_formu =  temp_formu + formu
#                 else:
#                     temp_formu =  temp_formu +r"\wedge"+ formu
#         Formula += temp_formu
#     return Formula

def new_get_formula_v1(save_dir, Dataset_name, i_class, k_run):

    # with open(save_dir+'W_best_{i}.pkl'.format(i=Dataset_name), 'rb') as f:
    #     W1s, Wcss, Wdss, a, b, t1, t2, Spatial, final_accuracy = pickle.load(f)
    with open(save_dir+'\W_best_'+Dataset_name+'_{i}_{k}.pkl'.format(i=i_class, k=k_run), 'rb') as f:
        W1s, Wcss, Wdss, a, b, t1, t2, Spatial, final_accuracy = pickle.load(f)

    W1s = W1s.detach().numpy()
    Wcss = Wcss.numpy()
    Wdss = Wdss.numpy()
    a = a.detach().numpy()
    b = b.detach().numpy()
    b = np.around(b, 3)
    t1 = t1.detach().numpy()
    t1 = np.around(t1, 1)
    t2 = t2.detach().numpy()
    t2 = np.around(t2, 1)
    final_accuracy = np.around(final_accuracy, 4)
    dim = np.shape(a)[-1]


    Formula = r"\quad"
    for i in range(len(Wcss)):
        temp_formu = r"\quad"

        for j in range(len(b)):
            if Wcss[i, j] == 1:
                # if sum(a)!=0:
                variable = ""
                first_occurrence_index = np.where(W1s[j,:] == 1)[0][0] 
                last_occurrence_index = np.where(W1s[j,:] == 1)[0][-1] 
                for k in range(dim):
                    if a[j, k]==1:
                        variable += "+x_{k}".format(k=k)
                    elif a[j, k]==-1:
                        variable += "-x_{k}".format(k=k)
                variable1 = "({z})".format(z=variable)

                formu = r"{g}[{t1}, {t2}] {x} > {b}".format(g=Spatial[j], t1=first_occurrence_index, 
                                                            t2=last_occurrence_index, x = variable1, b=b[j, 0])
                if len(temp_formu)<8:
                    temp_formu =  temp_formu + formu
                else:
                    temp_formu =  temp_formu +r"\wedge"+ formu
        Formula = Formula + r"\vee"+ temp_formu
    return Formula
    
def formula_plot(se_label, modelname, Dataset_name):
    import matplotlib.pyplot as plt

    with open('W_best_{i}.pkl'.format(i=Dataset_name), 'rb') as f:
        W1s, Wcss, Wdss, a, b, t1, t2, Spatial, final_accuracy = pickle.load(f)

    W1s = W1s.detach().numpy()
    # print(W1s.shape)
    Wcss = Wcss.numpy()
    Wdss = Wdss.numpy()
    a = a.detach().numpy()
    b = b.detach().numpy()
    b = np.around(b, 3)
    t1 = t1.detach().numpy()
    t1 = np.around(t1, 1)
    t2 = t2.detach().numpy()
    t2 = np.around(t2, 1)
    # final_accuracy = final_accuracy.numpy()
    final_accuracy = np.around(final_accuracy, 4)
    # print(np.shape(a))
    # print(np.shape(Wcss))
    dim = np.shape(a)[-1]

    fig, ax = plt.subplots()

    fig.set_size_inches(20, 6)

    Formula = r"$\quad$"
    for i in range(len(Wcss)):
        temp_formu = r"$\quad$"

        for j in range(len(b)):
            if Wcss[i, j] == 1:
                # if sum(a)!=0:
                variable = ""
                for k in range(dim):
                    if a[j, i, k]==1:
                        variable += "+x_{k}".format(k=k)
                    elif a[j, i, k]==-1:
                        variable += "-x_{k}".format(k=k)
                variable1 = "({z})".format(z=variable)
                # print(Spatial[j],t1[j, i],t2[j, i], j, b[j, i], variable1)
                formu = r"${g}[{t1}, {t2}] {x} > {b}$".format(g=Spatial[j], t1=t1[j, i], t2=t2[j, i], x = variable1, b=b[j, i])
                if len(temp_formu)<8:
                    temp_formu =  temp_formu + formu
                else:
                    temp_formu =  temp_formu +r"$\wedge$"+ formu
        Formula += temp_formu

    ax.axis('off')
    # tex =r"$\quad$"+r"$G_{[1, 2]} x > 1.1 \wedge $"+r"$G_{[1, 2]} x > 1.1 \wedge $"
    ax.text(-0.1, 0.6, Formula, fontsize=14, fontname='Times New Roman')
    print(Formula)
    Plot_final_accuracy = r"$The \quad Maximum \quad Testing  \quad Accuracy={z} \%$".format(z=final_accuracy)
    ax.text(0.1, 0.4, Plot_final_accuracy, fontsize=20, fontname='Times New Roman')

    Plot_Dataset_name = r"$Dataset \quad name={z}$".format(z=Dataset_name)
    ax.text(0.1, 0.2, Plot_Dataset_name, fontsize=20, fontname='Times New Roman')

    # print(Dataset_name)
    save_fig_name = "./MSTLNN_v1/results/{i}_{modelname}_{se_label}_{j}.jpg".format(i=Dataset_name, modelname=modelname, se_label=se_label, j=final_accuracy)
    plt.savefig(save_fig_name)
    plt.show()



# def new_formula_plot(se_label, modelname, Dataset_name):
#     import matplotlib.pyplot as plt

#     with open('W_best_{i}.pkl'.format(i=Dataset_name), 'rb') as f:
#         W1s, Wcss, Wdss, a, b, t1, t2, Spatial, final_accuracy = pickle.load(f)

#     W1s = W1s.detach().numpy()

#     Wcss = Wcss.numpy()
#     Wdss = Wdss.numpy()
#     a = a.detach().numpy()
#     b = b.detach().numpy()
#     b = np.around(b, 3)
#     t1 = t1.detach().numpy()
#     t1 = np.around(t1, 1)
#     t2 = t2.detach().numpy()
#     t2 = np.around(t2, 1)

#     final_accuracy = np.around(final_accuracy, 4)

#     dim = np.shape(a)[2]

#     fig, ax = plt.subplots()

#     fig.set_size_inches(12, 6)

#     Formula = []
#     for i in range(len(Wcss)):
#         for j in range(len(b)):
#             temp_formu = r"$\quad$"
#             if Wcss[i, j] == 1:
#                 # if sum(a)!=0:
#                 variable = ""
#                 first_occurrence_index = np.where(W1s[j,:] == 1)[0][0] 
#                 last_occurrence_index = np.where(W1s[j,:] == 1)[0][-1] 
#                 # print(t1[j, i], t2[j, i], first_occurrence_index, last_occurrence_index)
#                 # print(first_occurrence_index, last_occurrence_index)
#                 for k in range(dim):
#                     if a[j, i, k]==1:
#                         variable += "+x_{{{k}}}".format(k=k)
#                     elif a[j, i, k]==-1:
#                         variable += "-x_{{{k}}}".format(k=k)
#                 variable1 = "({z})".format(z=variable)

#                 # print(Spatial[j], t1[j, i],t2[j, i], j, b[j, i], variable1)
#                 formu = r"${g}_{{[{t1}, {t2}]}} {x} > {b}$".format(g=Spatial[j], t1=first_occurrence_index, 
#                                                                    t2=last_occurrence_index, x = variable1, b=b[j, i])
#                 if len(Formula)==0:
#                     temp_formu =  temp_formu + formu
#                 else:
#                     temp_formu =  temp_formu +r"$\wedge$"+ formu
  
#                 Formula.append(temp_formu)
    
#     ax.axis('off')
#     # tex =r"$\quad$"+r"$G_{[1, 2]} x > 1.1 \wedge $"+r"$G_{[1, 2]} x > 1.1 \wedge $"
#     start = 0.9 
#     print(len(Formula))
#     for z in range(len(Formula)):
        
#         ax.text(-0.1, start, Formula[z], fontsize=16, fontname='Times New Roman') 
#         start = start-(z+1)*0.05
#     # print(Formula)
    
#     final_accuracy = r"$The \quad Maximum \quad Testing  \quad Accuracy={z} \%$".format(z=final_accuracy)
#     print(Formula)
#     ax.text(0.1, 0.4, final_accuracy, fontsize=20, fontname='Times New Roman')

#     Dataset_name = r"$Dataset \quad name={z}$".format(z=Dataset_name)
#     ax.text(0.1, 0.2, Dataset_name, fontsize=20, fontname='Times New Roman')

#     # save_fig_name = "./MSTLNN_v1/results/{i}_{modelname}_{se_label}_{j}.jpg".format(i=Dataset_name, modelname=modelname, se_label=se_label, j=final_accuracy)
#     # plt.savefig(save_fig_name)

#     plt.show()

# formula_plot(0,Dataset_name=name)

