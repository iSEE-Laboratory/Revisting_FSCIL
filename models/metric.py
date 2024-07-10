import numpy as np


def generalised_avg_acc(alpha, acc):
    """
    param
    alpha: the range of alpha, e.g. [0,1,2,3,...,12]
    acc: all accuracies of each tasks, e.g [[80],[70, 65], [65, 60, 55]]..., element at position (i,j)
        indicates the accuracy of j-th task using model after trained on i-th task.
    """
    k = [len(i) - 1 for i in acc]
    res_matrix = []
    all_gen_acg_acc_different_alpha = []
    for a in alpha:
        res_matrix.append([])
        for i in range(len(acc)):
            acc0 = acc[i][0]
            if k[i] > 0:
                acc_new = acc[i][1:]
                g_avg_acc = (a * acc0 + sum(acc_new)) / (k[i] + a)
            else:
                g_avg_acc = acc0 if a != 0 else 0
            res_matrix[-1].append(g_avg_acc)
        all_gen_avg_acc = sum(res_matrix[-1]) / len(acc)
        all_gen_acg_acc_different_alpha.append(all_gen_avg_acc)
    return all_gen_acg_acc_different_alpha


def get_gacc(ratio, all_acc):
    alpha = [i for i in range(ratio + 1)]
    g_acc = generalised_avg_acc(alpha, all_acc)
    area = np.trapz(g_acc, x=alpha) / ((alpha[-1] - alpha[0]) * 100)
    return g_acc, area
