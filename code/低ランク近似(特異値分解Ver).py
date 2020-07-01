#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
from numpy.linalg import svd, matrix_rank
from PIL import Image
import sys

def low_rank_approximation(k : int, limit : int , A : np.ndarray):
    """
    特異値分解から低ランク近似まで
    k : 近似サイズ
    limit : ファイルサイズ
    A : 行列
    """
    u, s, v = svd(A)
    if(k > s.shape[0]):
        sys.exit()
    u = u[:,0:k]
    s = s[0:k]
    v = v[0:k]
    A_return = (u @ np.diag(s) @ v)
    return Image.fromarray(np.uint8(A_return))

def main():
    """
    project -> code ->  R9_lowrank.py
           |
            -> data ->  Airport.bmp
           |
            -> output -> out0000.jpg
    """
    data_path = '../data/'
    output_path = '../output/'
    file_name = 'Airport.bmp'
    file_size = 1024
    k = 1

    img = np.array(Image.open(data_path+file_name).convert('L'))
    low_rank_approximation( k, file_size, img).save(output_path+'out'+str(k).zfill(4)+'.jpg')

if __name__ == '__main__':
    main()

