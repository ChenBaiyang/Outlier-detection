import os, sys, time
import time
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import precision_recall_fscore_support
from models import MFIOD
from config import best_lambs, t_s

if __name__ == "__main__":
    dataset_list = np.array(os.listdir('data'))
    for fname in dataset_list:
        dataset = fname.split('\\')[-1].split('.')[0]
        print('Working on ', fname)
        mat = loadmat('data/'+fname)
        data = np.array(mat['trandata'])[:, 0:-1]
        label = np.array(mat['trandata'], dtype=np.int32)[:, -1]
        label_list, label_counts = np.unique(label, return_counts=True)
        pos_label = label_list[np.argmin(label_counts)]
        groundtruth = (label == pos_label)
        print("Total samples:{}  Outlier counts:{}".format(len(label), sum(groundtruth)))
        print("Data shape:{}".format(data.shape))

        data = minmax_scale(data)
        lambs = best_lambs[fname]
        start = time.time()
        print('lambda:', lambs)
        mvgod = MFIOD(data, lambs=lambs)
        order = np.argsort(mvgod)

        results = []
        print('---------------------------------')
        print('t    \tPrecision \tRecal')
        print('---------------------------------')
        for t in t_s[fname]:
            pos_idx = order[-t:]
            result = np.zeros_like(groundtruth)
            result[pos_idx] = True
            P, R, F1, _ = precision_recall_fscore_support(groundtruth, result, average='binary')
            print('{:5s}\t{:.2f} \t{:.2f}'.format(str(t), P*100, R*100))
            results.append([P, R])

        results = np.array(results)
        ave_P, ave_R = results.mean(axis=0)
        print('ave  \t{:.2f} \t{:.2f}'.format(ave_P*100, ave_R*100))
        results = np.append(results, [[ave_P,ave_R]], axis=0)
        Pt, Rt = results[:, 0].tolist(), results[:, 1].tolist()
        end = time.time()
        print("time = {:.4f}".format(end - start))

        # Save results
        Pt = ['{:.3f}'.format(i*100) for i in Pt]
        Rt = ['{:.3f}'.format(i*100) for i in Rt]
        record = ','.join([dataset, 'MFIOD', 'Pt,'+','.join(Pt), 'Rt,'+','.join(Rt)])+'\n'
        open('results_MFIOD.csv', 'a').write(record)


