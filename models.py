import numpy as np

def ndarray_find(nd_data, num):
    out = []
    for i in range(len(nd_data)):
        if nd_data[i] == num:
            out.append(i)
    return np.array(out)

def make_relation_matrix(x, vector, ek):
    dif = np.abs(x-vector)
    result = 1 - dif
    big = dif > ek + 1e-6
    result[big] = 0
    return result

def MFIOD(data, lambs):
    n, m = data.shape
    n_lamb = len(lambs)
    M_R_ck = np.zeros((n_lamb, m, n, n), dtype=np.float32)
    for l, lamb in enumerate(lambs):
        for i in range(0, m):
            for j in range(0, n):
                M_R_ck[l, i, j] = make_relation_matrix(data[j, i], data[:, i], lamb)
    M_R_ck = np.mean(M_R_ck, axis=0)

    LA = np.arange(m)
    weight = np.zeros((n, m), dtype=np.float32)
    Acc_Appr = np.zeros((n, m), dtype=np.float32)
    for l in range(m):
        A_d = np.setdiff1d(LA, l)[0]
        M_R_ck_l, ia, ic = np.unique(M_R_ck[l], axis=0, return_index=True, return_inverse=True)
        for i in range(0, M_R_ck_l.shape[0]):
            i_tem = ndarray_find(ic, i)
            M_R_P = M_R_ck[A_d]
            M_R_B = np.tile(M_R_ck_l[i,:], (n, 1))
            Low_Appr = sum(np.min(np.maximum(1 - M_R_P, M_R_B), axis=1))
            Upp_Appr = sum(np.max(np.minimum(M_R_P, M_R_B), axis=1))
            Acc_Appr[i_tem, l] = Low_Appr / Upp_Appr
            weight[i_tem, l] = M_R_ck_l[i, :].sum() / n

    OF = 1 - Acc_Appr * weight
    MSOD = np.mean(OF * (1-np.power(weight, 1/3)), axis=1)
    return MSOD


if __name__ == "__main__":
    pass