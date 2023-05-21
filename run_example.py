import os, sys, time
import time
import numpy as np
import numpy.matlib
from sklearn.preprocessing import minmax_scale
from utils import ufrs_kersim_bak, matrix_rewrite, ndarray_find

def MFIOD(data, lambs):
    n, m = data.shape
    n_lamb = len(lambs)
    M_R_ck = np.zeros((n_lamb, m, n, n), dtype=np.float32)
    for l, lamb in enumerate(lambs):
        for i in range(0, m):
            for j in range(0, n):
                for k in range(0, j+1):
                    M_R_ck[l, i, j, k] = ufrs_kersim_bak(data[j, i], data[k, i], lamb)
                    M_R_ck[l, i, k, j] = M_R_ck[l, i, j, k]
    M_R_ck = np.mean(M_R_ck, axis=0)
    print(M_R_ck)

    LA = np.arange(m)
    weight1 = np.zeros((n, 2))
    weight3 = np.zeros((n, 2))
    Acc_A_a = np.zeros((2, n), dtype=np.float32)
    for l in [1,0]:
        A_d = np.setdiff1d(LA, l)[0]
        M_R_ck_l, ia, ic = np.unique(M_R_ck[l], axis=0, return_index=True, return_inverse=True)
        for i in range(0, M_R_ck_l.shape[0]):
            i_tem = ndarray_find(ic, i)
            M_R_P = M_R_ck[A_d]
            M_R_B = np.matlib.repmat(M_R_ck_l[i,:], n, 1)
            Low_A = sum(np.min(np.maximum(1 - M_R_P, M_R_B), axis=1))
            Upp_A = sum(np.max(np.minimum(M_R_P, M_R_B), axis=1))
            Acc_A_a[l, i_tem] = Low_A / Upp_A
            print(l,i_tem,Low_A,Upp_A, Low_A / Upp_A)
            print(lamb, l, i_tem, 'Low_app:', np.min(np.maximum(1 - M_R_P, M_R_B), axis=1))
            print(lamb, l, i_tem, 'Upp_app:', np.max(np.minimum(M_R_P, M_R_B), axis=1))

            weight3 = matrix_rewrite(weight3, i_tem, l, 1 - np.power(sum(M_R_ck_l[i, :])/n, 1/3))
            weight1 = matrix_rewrite(weight1, i_tem, l, (sum(M_R_ck_l[i, :])/n))

    GOD = np.zeros((n, 2), dtype=np.float32)
    for i in range(0, n):
        for k in range(0, 2):
            GOD[i, k] = 1 - Acc_A_a[k, i] * weight1[i, k]
            print(i,k,GOD[i,k],weight1[i,k], weight3[i,k])

    OD = np.mean(GOD * weight3, axis=1)
    print(OD)
    assert len(OD) == n

    return OD

if __name__ == "__main__":
    data = np.array(
           [[0.7, 9],
            [0.3, 6],
            [0.5, 2],
            [0.2, 3],
            [0.4, 7],
            [0.6, 3]])

    data = minmax_scale(data)
    print(data)

    OD = MFIOD(data, lambs=[0.2, 0.3])
    print(OD)

# OUTPUTS:
# [[1.         1.         0.5       ]
#  [0.2        0.57142857 0.        ]
#  [0.6        0.         1.        ]
#  [0.         0.14285714 0.        ]
#  [0.4        0.71428571 0.        ]
#  [0.8        0.14285714 1.        ]]
# [[[1.         0.         0.         0.         0.         0.8       ]
#   [0.         1.         0.         0.8        0.8        0.        ]
#   [0.         0.         1.         0.         0.8        0.8       ]
#   [0.         0.8        0.         1.         0.         0.        ]
#   [0.         0.8        0.8        0.         1.         0.        ]
#   [0.8        0.         0.8        0.         0.         1.        ]]
#
#  [[1.         0.         0.         0.         0.35714287 0.        ]
#   [0.         1.         0.         0.         0.85714287 0.        ]
#   [0.         0.         1.         0.85714287 0.         0.85714287]
#   [0.         0.         0.85714287 1.         0.         1.        ]
#   [0.35714287 0.85714287 0.         0.         1.         0.        ]
#   [0.         0.         0.85714287 1.         0.         1.        ]]
#
#  [[1.         0.         0.         0.         0.         0.        ]
#   [0.         1.         0.         1.         1.         0.        ]
#   [0.         0.         1.         0.         0.         1.        ]
#   [0.         1.         0.         1.         1.         0.        ]
#   [0.         1.         0.         1.         1.         0.        ]
#   [0.         0.         1.         0.         0.         1.        ]]]
# 1 [3 5] 0.11413042701865309
# 1 [2] 0.11731842755043598
# 1 [1] 0.11570247135953127
# 1 [4] 0.1438356061621755
# 1 [0] 0.1393034724663286
# 0 [2] 0.06116207521854913
# 0 [3] 0.03205127990819922
# 0 [4] 0.34365324573500217
# 0 [1] 0.3126934956125806
# 0 [5] 0.195729531712633
# 0 [0] 0.1711026573141858
# 0 0 0.9486692 0.3000000019868215 0.33056704844000406
# 0 1 0.9684909 0.2261904776096344 0.39070898674713306
# 1 0 0.86449945 0.4333333373069763 0.24327043747048216
# 1 1 0.9641873 0.3095238109429677 0.3235567699766094
# 2 0 0.97349644 0.4333333373069763 0.24327043747048216
# 2 1 0.94692737 0.4523809552192688 0.23234142689093684
# 3 0 0.99038464 0.3000000019868215 0.33056704844000406
# 3 1 0.9456522 0.4761904776096344 0.21910333360516887
# 4 0 0.8510836 0.4333333373069763 0.24327043747048216
# 4 1 0.94691783 0.3690476218859355 0.28271105568791277
# 5 0 0.91518384 0.4333333373069763 0.24327043747048216
# 5 1 0.9456522 0.4761904776096344 0.21910333360516887
# [0.34599844 0.26113825 0.22841668 0.26729204 0.23737381 0.21491636]
# [0.34599844 0.26113825 0.22841668 0.26729204 0.23737381 0.21491636]