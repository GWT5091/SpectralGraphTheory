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
        if(len(row) == 3):
            A[row[0]-1, row[1]-1] = row[2]
        elif(len(row) == 2):
            A[row[0]-1, row[1]-1] = 1
    #次数行列作成
    D = np.zeros((n_val, n_val))
    for i in range(len(A)):
        D[i,i] = A[i].sum()
        
    #ラプラシアン行列作成
    L = ((-1)*A) + D
    
    return A, D, L

def calc_edge(val_list:list, A:np.ndarray)->int:
    """
    2つの部分ネットワークをまたぐリンクの集合の数を計算
    val_list : 部分ネットワークの頂点Index（どちらでも可）
    A : 隣接行列
    """
    edge_val = 0
    for i in val_list:
        for num, j in enumerate(A[i]):
            if not num in val_list:
                if j > 0:
                    edge_val += 1
    return edge_val

def calc_r_val(edge_val:int,vertex_sum,part_of_net_vertex:int):
    """
    カット比の計算
    edge_val:2つの部分ネットワークをまたぐリンクの集合の数
    vertex_sum : グラフの総頂点数
    part_of_net_vertex:現在選択中の部分グラフにおける頂点数
    """
    return edge_val / ((vertex_sum - part_of_net_vertex) * part_of_net_vertex)

def calc_min_cut_ratio(a:np.ndarray, vec_list:list, r_min = 100):
    """
    最小カット比計算
    A : 隣接行列
    vec_list:第2最小固有ベクトル
    """
    vec_df = pd.DataFrame(data=vec_list,columns=['data']).sort_values('data', ascending=False)
    s_list = list()
    tmp_list = list()
    sum_v = len(vec_df.index)
    edge = 0

    for i in vec_df.index[0:sum_v-1]:
        tmp_list.append(i)
        edge = calc_edge(tmp_list,a)
        r_val = calc_r_val(edge, sum_v, len(tmp_list))
        if(r_val <= r_min):
            r_min = r_val
            s_list = tmp_list.copy()
    return r_min, s_list

def main():
    """
    project -> code ->  R10_graphcut.py
           |
            -> data ->  07_network.txt
           |
            -> output -> R10_graphcut_output.txt
    """
    file_path = "../data/"
    file_name = "07_network.txt"
    output_path = "../output/"
    out_put_name = "R10_graphcut_output.txt"
    data_list = list()


    with open(file_path+file_name) as f:
        for read_line in f:
            data_list.append([int(i) for i in read_line.replace('\n','').split(' ')])

    a, d, l = cal_matrix(data_list,max(max(data_list)))
    lam, vec = LA.eig(l)
    small_vec_list = list()
    for i in range(len(vec)):
        small_vec_list.append(vec[i][1])
    min_r ,min_vertex_list = calc_min_cut_ratio(a,small_vec_list)
    
    min_vertex_list.sort()
    out_put_shape = [str(min_r), " ".join([str(i+1) for i in min_vertex_list])]

    with open(output_path+out_put_name, mode='w') as f:
        f.write('\n'.join(out_put_shape))

if __name__ == "__main__":
    main()

