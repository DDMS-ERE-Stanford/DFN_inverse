#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 21:59:16 2021

@author: zitongzhou
"""

import numpy as np
import os
import pickle as pkl
from tqdm import tqdm
import time
import natsort
import shutil


## Pickle CDFs into a file for each of the seed.
# def pk_cdf(dir):
#     os.chdir(dir)
#     seeds_list = natsort.natsorted(os.listdir(dir+'/')) #sort vtk files by the number
#     seeds_list = [i for i in seeds_list if i.startswith('OutputSimu')]
#     for seeds_folder in seeds_list:
#         os.chdir(dir+'/'+seeds_folder+'/cdf/')
#         index = []
#         cdf = []
#         # seed_ind = seeds_folder[-2:]
#         cdf_list = os.listdir(os.getcwd())
#         for f_i in tqdm(range(len(cdf_list)), unit="files"):
#             filename = cdf_list[f_i]
#             if filename.endswith(".txt"):
#                 index_s = [i for i in filename if i.isnumeric()]
#                 index.append(int("".join(index_s)))
#                 lines = np.loadtxt(fname = filename)
#                 cdf.append(lines)
#         os.chdir(dir+'/'+seeds_folder+'/')    
#         with open ('ind_cdf.pkl', 'wb') as file:
#             pkl.dump([index, cdf], file)
#         os.chdir(dir)
    
#     return
            
## Pickle PDFs into a file for each seed.
def pk_pdf(dir):
    os.chdir(dir)
    seeds_list = natsort.natsorted(os.listdir(dir+'/')) #sort vtk files by the number
    seeds_list = [i for i in seeds_list if i.startswith('OutputSimu')]
    for seeds_folder in seeds_list:
        index = []
        pdf = []
        os.chdir(dir+'/'+seeds_folder+'/pdf/')

        # seed_ind = seeds_folder[-2:]
        pdf_list = os.listdir(os.getcwd())
        for f_i in tqdm(range(len(pdf_list)), unit="files"):
            filename = pdf_list[f_i]
            if filename.endswith(".txt"):
                index_s = [i for i in filename if i.isnumeric()]
                ## there is pdf10001.txt in the folder, it should be ignored.
                if int("".join(index_s)) <= 10000:
                    index.append(int("".join(index_s)))
                    lines = np.loadtxt(fname = filename)
                    pdf.append(lines)
        os.chdir(dir+'/'+seeds_folder+'/')    
        with open ('ind_pdf.pkl', 'wb') as file:
            pkl.dump([index, pdf], file)
        os.chdir(dir)
    
    return
    
def count_pdf(dir):
    os.chdir(dir)
    seeds_list = natsort.natsorted(os.listdir(dir+'/')) #sort vtk files by the number
    seeds_list = [i for i in seeds_list if i.startswith('OutputSimu')]
    dict_count = {}
    for seeds_folder in seeds_list:
        os.chdir(dir+'/'+seeds_folder+'/pdf/')
        dict_count[seeds_folder[16:]] =  len(os.listdir())
        
    return dict_count

def rm_cdf(dir):
    seeds_list = natsort.natsorted(os.listdir(dir+'/')) #sort vtk files by the number
    seeds_list = [i for i in seeds_list if i.startswith('OutputSimu')]
    for seeds_folder in seeds_list:

        os.chdir(dir+'/'+seeds_folder+'/')
        if os.path.isdir('cdf'):
            shutil.rmtree('cdf', ignore_errors=True)
        if os.path.isdir('dfn'):
            shutil.rmtree('dfn', ignore_errors=True)
        if os.path.isdir('pdf'):
            shutil.rmtree('pdf', ignore_errors=True)        
        
if __name__ == '__main__':
    # dirs = [
    #     '/Users/zitongzhou/Desktop/2021-03-02.tmp/Correlated_100part',
    #     '/Users/zitongzhou/Desktop/2021-03-02.tmp/Correlated_1000part',
    #     '/Users/zitongzhou/Desktop/2021-03-02.tmp/Uncorrelated_100part',
    #     '/Users/zitongzhou/Desktop/2021-03-02.tmp/Uncorrelated_1000part',
    #     ]
    dirs = [
        '/Users/zitongzhou/Desktop/2021-04-16.tmp/Uncorrelated_100part',
        '/Users/zitongzhou/Desktop/2021-04-16.tmp/Uncorrelated_1000part']
    start = time.time()
    for dir in dirs:
        # print(count_pdf(dir))
        pk_pdf(dir)
        # rm_cdf(dir)
    end = time.time()
    print('Processing all files took ', end-start, ' seconds.')
    
    
    


