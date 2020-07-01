#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.linalg as LA
import pandas as pd

def cal_matrix(data:list,n_val:int):
    """
    A：隣接行列
    D：次数行列
    L：ラプラシアン行列
    を作成する
    data:各要素が[ノードID1,ノードID2,エッジ重み]となっているリスト
    n_val:ノードの総数
    """
    #隣接行列作成
    A = np.zeros((n_val,n_val))
    for row in data:
        A[row[0]-1, row[1]-1] = row[2]
        
    #次数行列作成
    D = np.zeros((n_val, n_val))
    for i in range(len(A)):
        D[i,i] = A[i].sum()
        
    #ラプラシアン行列作成
    L = ((-1)*A) + D
    
    return A, D, L

def get_small_index(val_array):
    """
    numpyの固有値と固有ベクトルは昇順になっているのでいらないが一応
    val_array:固有値
    """
    df = pd.DataFrame(val_array).sort_values(0)
    return list(df.index)

def plot_mat_3D(X,Y,Z):
    """
    画像提出用
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


    ax.plot(X,Y,Z,marker="o",linestyle='None')

    plt.show()
    fig.savefig("../output/C20.jpg")

def main():
    """
    project -> code ->  R7_drawing.py
           |
            -> data ->  graph.txt
           |
            -> output -> C20.jpg
    """
    file_path = "../data/"
    file_name = "graph.txt"
    data_list = list()

    with open(file_path+file_name) as f:
        for read_line in f:
            data_list.append([int(i) for i in read_line.replace('\n','').split(' ')])

    #総ノード数
    n = max(data_list)[0]
    A, D, L = cal_matrix(data_list,n)
    lam, vec = LA.eigh(L)
    small_sort_index = get_small_index(lam)

    #提出用ノードIDとxyz座標出力
    for i in range(len(vec)):
        print("ノードID:"+str(i+1)+", x:"+str(vec[small_sort_index[1],i])+", y:"+str(vec[small_sort_index[2],i])+", z:"+str(vec[small_sort_index[3],i]))
    
    plot_mat_3D(vec[small_sort_index[1]],vec[small_sort_index[2]],vec[small_sort_index[3]])

if __name__ == "__main__":
    main()

