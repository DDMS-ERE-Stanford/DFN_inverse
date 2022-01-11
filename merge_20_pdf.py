#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 21:08:18 2021

@author: ztzhou
Merge sort the PDFs to combine the 20 seeds(realizations) outputs together 
and then take the average.
pdf_simu['pdf6'][seed3]: pdf 6 for seed 3;
pdf_simu['pdf6']['mean']: the combined pdf6 from all seeds;
"""

import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt
import natsort
import time
from tqdm import tqdm
import scipy.interpolate
import matplotlib.ticker as mticker

def sort_merge(pdf1, pdf2):
    '''pdf1,2: (x, 2) arrays representing pdfs, this algorithm merge pdf1 and 
    pdf2 by combining 2 arrays with x sorted.'''
    new_pdf = []
    i_1 = 0
    i_2 = 0
    while i_1 <len(pdf1) and i_2 < len(pdf2):
        temp_1, temp_2 = pdf1[i_1][0], pdf2[i_2][0]
        if temp_1 < temp_2:
            if len(new_pdf)>0 and temp_1 == new_pdf[-1][0]:
                new_pdf[-1][1] += pdf1[i_1][1]
            else:
                new_pdf.append(pdf1[i_1])
            i_1 += 1
        elif temp_1 == temp_2:
            if len(new_pdf)>0 and temp_1 == new_pdf[-1][0]:
                new_pdf[-1][1] += pdf1[i_1][1] + pdf2[i_2][1]
            else:
                new_pdf.append([pdf1[i_1][0], pdf1[i_1][1] + pdf2[i_2][1]])
            i_1 += 1
            i_2 += 1
        else:
            if len(new_pdf)>0 and temp_2 == new_pdf[-1][0]:
                new_pdf[-1][1] += pdf2[i_2][1]
            else:
                new_pdf.append(pdf2[i_2])
            i_2 += 1
    while i_1 < len(pdf1):
        if len(new_pdf)>0 and pdf1[i_1][0] == new_pdf[-1][0]:
            new_pdf[-1][1] += pdf1[i_1][1]
        else:
            new_pdf.append(pdf1[i_1])
        i_1 += 1
    while i_2 < len(pdf2):
        if len(new_pdf)>0 and pdf2[i_2][0] == new_pdf[-1][0]:
            new_pdf[-1][1] += pdf2[i_2][1]
        else:
            new_pdf.append(pdf2[i_2])
        i_2 += 1
    return new_pdf
    

def pdf2cdf(pdf):
    '''Normalize the pdf to sum to 1, then turn into cdf'''
    cdf = pdf.copy()
    cdf[:,1] = cdf[:,1]/sum(cdf[:,1])
    for i in range(len(cdf)-1):
        cdf[i+1,1] += cdf[i, 1]

    return cdf

def normalize(pdf):
    norm_pdf = pdf.copy()
    norm_pdf[:, 1] = norm_pdf[:,1]/np.sum(norm_pdf[:,1])
    return norm_pdf

def merge_pdf(dir='', filename='combined_seeds', seed_n=20,):
    pdf_simu = {'pdf'+str(i):{} for i in range(1,10001)}
    os.chdir(dir)
    seeds_list = natsort.natsorted(os.listdir(dir+'/')) #sort seeds by the number
    seeds_list = [i for i in seeds_list if i.startswith('OutputSimu')]
    for f_i in tqdm(range(len(seeds_list)), unit="seeds"):
        seeds_folder = seeds_list[f_i]
        seed_ind = seeds_folder[-2:]
        ##pickle load all pdfs for 1 seed, with the index referring to the CD pair.
        with open (seeds_folder+'/ind_pdf.pkl', 'rb') as file:
            [index, pdf] = pkl.load(file)
        ##update pdf_simu[pdfn][seedsm] for n in index
        for i,ind in enumerate(index):
            pdf_simu['pdf'+str(ind)]['seed'+str(seed_ind)] = pdf[i]
            if 'mean' not in pdf_simu['pdf'+str(ind)].keys():
                pdf_simu['pdf'+str(ind)]['mean']= []
            pdf_simu['pdf'+str(ind)]['mean'] = sort_merge(
                pdf_simu['pdf'+str(ind)]['mean'], pdf[i]
                )
    for pdf_ind in pdf_simu.keys():
        pdf_simu[pdf_ind]['mean'] = normalize(np.array(
            pdf_simu[pdf_ind]['mean'])
            )     
    with open(filename, 'wb') as file:
        pkl.dump([pdf_simu], file)
        
    return pdf_simu

def pdf2ICDF(pdf, x=np.round(np.linspace(0.01, 0.99, 50), 2)):
    '''Convert pdf array to inverse cdf(ICDF)'''
    pdf = np.array([p for p in pdf if p[1]!=0])
    cdf = pdf2cdf(pdf)
    y_interp = scipy.interpolate.interp1d(
        cdf[:,1], cdf[:,0], fill_value="extrapolate"
        )
    y = y_interp(x)
    return y


if __name__ == '__main__':
    '''merge pdf for each simulation'''
    dirs = [
        '/Users/zitongzhou/Desktop/2021-04-16.tmp/Uncorrelated_100part',
        '/Users/zitongzhou/Desktop/2021-04-16.tmp/Uncorrelated_1000part']
    filenames = [
        # 'corr_100_20pdf.pkl',
        # 'corr_1000_20pdf.pkl',
        'uncorr_100_20pdf.pkl',
        'uncorr_1000_20pdf.pkl',
        ]
    # start = time.time()
    # for dir_simu, filename in zip(dirs, filenames):
    #     pdf_simu = merge_pdf(dir=dir_simu, filename=filename, )
    # end = time.time()
    # print('Combining seeds for {} folders took '.format(len(dirs)), end-start, ' seconds.')
    # Combining seeds for 4 folders took  689.9622631072998  seconds.
    '''pickle C,D and ICDF for uncorrelated simulations, 100 particles'''
    input_filename = '/Users/zitongzhou/Desktop/FRACTURE/seeds20/C_d.txt'
    input_CD = np.loadtxt(fname = input_filename)
    
    ##compute the minumal number of connected cases for each file
    n_conn = {}
    
    '''Randomly sample 6 simulations, 
    from uncorrelated 100 and 1000 dataset for further comparison'''
    pdf_samples = {}
    np.random.seed(88)
    rand_i = np.random.randint(1, 10001, 6)
    for ind,f in enumerate(filenames):
        os.chdir(dirs[ind])
        pdf_samples[f] = {}
        with open(f, 'rb') as file:
            [pdf_simu] = pkl.load(file)
            n_conn[f] = {i: len(pdf_simu[i].keys())-1 for i in pdf_simu.keys()}
        for i in rand_i:
            assert pdf_simu['pdf'+str(i)]['mean'] is not None, \
                '{} is not in {}'.format(i, f)
            pdf_samples[f][str(i)] = pdf_simu['pdf'+str(i)]
    
    # with open('uncorr_100_20pdf_conn.pkl', 'wb') as file:
    #     pkl.dump([C_conn, D_conn, num_conn], file)
    for i in n_conn.keys():
        print(
            i+':', min(list(n_conn[i].values()))
        )
        
        
    ''' plot the histogram of the connected cases for each simulation'''
    os.chdir('/Users/zitongzhou/Desktop/FRACTURE/images')
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rc('font',**{'family':'serif','serif':['Times']})
    plt.rcParams['text.usetex'] = True
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    
    markerstyles = ['.', '1', '*', 'x','d']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    ''' plot the histogram of the connected cases for each simulation'''
    figsize = (3.5, 4)
    cols = 1
    rows = 1
    fig_cdf, ax_cdf = plt.subplots(rows, cols, figsize=figsize, 
                              constrained_layout=True, 
                             # sharey=True,
                             )
    
    figsize = (7, 4)
    cols = 2
    rows = 1
    fig, axs = plt.subplots(rows, cols, figsize=figsize, 
                              constrained_layout=True, 
                             # sharey=True,
                             )

    # fig2, axs2 = plt.subplots(rows, cols, figsize=figsize, 
    #                           # constrained_layout=True, 
    #                           # sharey=True,
    #                           )
    # axs2_cdf = axs2.twinx()
    axs_all = [axs.flat[0], axs.flat[1]]
               # , axs2.flat[0], axs2.flat[1]]
    titles = ['100 particles', '1000 particles', ]
              # '100 particles', '1000 particles']!
    suptitles = ['corr_simu_hist', 'uncorr_simu_hist']
    linestyles = ['-', '-.', ':', '--', '(3, 1, 1, 1, 1, 1)', 'dashdot']
    labels = ['100', '1000']
    for i, filename in enumerate(filenames):
        data = np.array(list(n_conn[filenames[i]].values()))
        d = np.diff(np.unique(data)).min()
        left_of_first_bin = data.min() - float(d)/2
        right_of_last_bin = data.max() + float(d)/2
        bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)
        n, bins, patches = axs_all[i].hist(
            data,
            bins=bins,
            ec='k',
        )
        
        # axs_cdf = axs_all[i].twinx()
        cdf_data = n.cumsum()
        cdf_data = cdf_data/cdf_data.max()
        cdf_bin = (bins[:-1]+bins[1:])//2
        # axs_cdf.plot(cdf_bin, cdf_data, color=colors[1])
        ax_cdf.plot(cdf_bin, cdf_data, linestyle = linestyles[i], 
                    color=colors[i], label=labels[i])
        axs_all[i].set_xlabel('connected cases in 20')
        axs_all[i].set_ylabel('count')
        # axs_cdf.set_ylabel('CDF')
        axs_all[i].locator_params(axis='x', integer=True)
        axs_all[i].set_ylim(bottom=-100, top=n.max()+1000)
        axs_all[i].set_xlim(left=left_of_first_bin)
        axs_all[i].set_xticks(np.arange(data.min(), data.max()+1))
        axs_all[i].set_title(titles[i])
        for m in range(len(n)):
            axs_all[i].text(bins[m]+0.2,n[m]+100,str(int(n[m])), 
                            rotation='vertical',)
    # fig.savefig(suptitles[0]+'.pdf')
    ax_cdf.legend(title='nb\_part')
    # fig_cdf.savefig('hist_cdf.pdf')
    # fig.savefig(suptitles[1]+'.pdf')
    
    ## distribution of connected cases
    fig, axs = plt.subplots(1, 2, figsize=(7, 4),  sharey='row', sharex='col')

    for f, ax in zip(filenames, axs.flat):
        C_conn = []
        D_conn = []
        num_conn = []
        for key in n_conn[f].keys():
            ind = int(''.join([i for i in key if i.isnumeric()]))
            C_conn.append(input_CD[ind-1,0])
            D_conn.append(input_CD[ind-1, 1])
            num_conn.append(n_conn[f][key])
        C_conn = np.array(C_conn)
        D_conn = np.array(D_conn)
        num_conn = np.array(num_conn)    
        c01map = ax.scatter(C_conn, D_conn, c=num_conn, cmap='jet', marker=",", s = 0.5,
                            # interpolation='nearest',
                  # extent=[C_d[:,0].min(), C_d[:,0].max(), C_d[:,1].min(), C_d[:,1].max()],
                  vmin=num_conn.min(), vmax = num_conn.max(),
                  # origin='lower',aspect='auto'
                  )
        ax.set_xlim(2.5, 6.5)
        ax.set_xticks(np.linspace(2.5, 6.5, 5, endpoint=True))
        ax.set_ylim(1.0, 1.3)
        ax.set_xlabel('C [-]')
        ax.set_ylabel('D [-]')
        v1 = np.arange(np.min(num_conn), np.max(num_conn)+1,)
        fig.colorbar(c01map, ax=ax, fraction=0.05, pad=0.04,ticks=v1,)


        # ax1.set_ylim([0, 12])
        # ax.set_xlabel('C [-]')
        # ax.set_ylabel('D [-]')
    plt.tight_layout()
    fig.savefig('conn_density.pdf', format='pdf',bbox_inches='tight')
    plt.show()
    # end2 = time.time()

    
    '''plot 2 figures, 1st for comparing 20 seeds for 6 simulations for 100
    particles, 2nd for comparing models with 100 particles and 1000 particles,
    3rd for comparing 20 seeds for 6 simulations for 1000 particles'''
    figsize = (10, 8)
    cols = 3
    rows = 2
    fig1, axs = plt.subplots(rows, cols, figsize=figsize, 
                             constrained_layout=True, 
                             # sharey='row', 
                             # sharex='col',
                             )
    axs = axs.flat
    
    fig2, axs2 = plt.subplots(rows, cols, figsize=figsize, 
                             constrained_layout=True, 
                             # sharey='row', 
                             # sharex='col',
                             )
    axs2 = axs2.flat
    
    fig3, axs3 = plt.subplots(rows, cols, figsize=figsize, 
                             constrained_layout=True, 
                             # sharey='row', 
                             # sharex='col',
                             )
    axs3 = axs3.flat
    
    # filename1 = filenames[2]
    # filename2 = filenames[3]
    filename1 = filenames[0]
    filename2 = filenames[1]

    for ind, ax, ax2, ax3 in zip(rand_i, axs, axs2, axs3):
        f1 = pdf_samples[filename1][str(ind)]
        f2 = pdf_samples[filename2][str(ind)]
        for key in f1.keys():
            if key != 'mean':
                pdf = f1[key]
                pdf = np.array([p for p in pdf if p[1]!=0])
                ax.plot(
                    np.log(pdf[:,0]), pdf2cdf(pdf)[:,1], 
                    label=''.join([i for i in key if i.isnumeric()]).zfill(2)
                    )
                
            else:
                pdf1 = f1[key]
                pdf2 = f2[key]
                ax2.plot(
                    np.log(pdf2[:,0]), pdf2cdf(pdf2)[:,1], 
                    '-', color=colors[0],linewidth=2.5,
                    label='1000'
                    )            
                ax2.plot(
                    np.log(pdf1[:,0]), pdf2cdf(pdf1)[:,1], 
                    '-.', color=colors[1],linewidth=2.5,
                    label='100'
                    )
        for key in f2.keys():
            if key != 'mean':
                pdf1000 = f2[key]
                pdf1000 = np.array([p for p in pdf1000 if p[1]!=0])
                ax3.plot(
                    np.log(pdf1000[:,0]), pdf2cdf(pdf1000)[:,1], 
                    label=''.join([i for i in key if i.isnumeric()]).zfill(2)
                    )
        # ax.legend(title='seed', loc='lower right', ncol=2)#, mode="expand")
        # ax3.legend(title='seed', loc='lower right', ncol=2)
        # ax2.legend(loc="lower right")#title='nb\_part=')
        ax2.legend().get_frame().set_edgecolor('b')
        ax2.legend().get_frame().set_linewidth(0.0)
        ax2.legend(loc="lower right")

        ax.set_title(
            '(C,D)=({:.2f}, {:.2f})'.format(input_CD[ind-1,0], input_CD[ind-1, 1])
            )
        ax3.set_title(
            '(C,D)=({:.2f}, {:.2f})'.format(input_CD[ind-1,0], input_CD[ind-1, 1])
            )
        ax2.set_title(
            '(C,D)=({:.2f}, {:.2f})'.format(input_CD[ind-1,0], input_CD[ind-1, 1])
            )
        # ax.set_xlabel('log(particle arrivial time)')
        # ax.set_ylabel('CDF')   
        ax.xaxis.set_major_locator(mticker.LinearLocator(6))
        ax.xaxis.set_major_formatter('{x:.00f}')
        ax.set_xlim([15,30])
        # ax2.set_xlabel('log(particle arrivial time)')
        # ax2.set_ylabel('CDF') 
        ax2.xaxis.set_major_locator(mticker.LinearLocator(6))
        ax2.xaxis.set_major_formatter('{x:.00f}')
        ax2.set_xlim([15,38])
        # ax3.set_xlabel('log(particle arrivial time)')
        # ax3.set_ylabel('CDF')   
        ax3.xaxis.set_major_locator(mticker.LinearLocator(6))
        ax3.xaxis.set_major_formatter('{x:.00f}')
        ax3.set_xlim([15,30])
    # fig1.savefig('uncorr20seeds.pdf')
    fig2.savefig('uncorr100_1000par.pdf')
    # fig3.savefig('uncorr20seeds_1000.pdf')
    
    '''plot a figure of mean cdf for 6 simulations, and the ICDF'''
    figsize = (7, 4)
    cols = 2
    rows = 1
    fig, axs = plt.subplots(rows, cols, figsize=figsize, 
                            constrained_layout=True)
    axs = axs.flat
    x = np.round(np.linspace(0.01, 0.99, 50), 2)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
              'tab:purple', 'tab:brown','tab:pink', 
              'tab:gray', 'tab:olive', 'tab:cyan']
    dashes_choices = [
        [],[1,1],[3,5,1, 5],[5, 10], [5, 1], [3, 1, 1, 1, 1, 1]
        ]
    for i,ind in enumerate(rand_i):
        f1 = pdf_samples[filename1][str(ind)]
        pdf = f1['mean']
        pdf = np.array([p for p in pdf if p[1]!=0])
        axs[0].plot(
            np.log(pdf[:,0]), pdf2cdf(pdf)[:,1], 
            color=colors[i], 
            linewidth = 2,
            dashes=dashes_choices[i],
            # linestyle = linestyles[i],
            label=
            '({:.2f}, {:.2f})'.format(input_CD[ind-1,0], input_CD[ind-1, 1])            
            )
        
        axs[1].plot(
            x, np.log(pdf2ICDF(pdf,x)), 
            color=colors[i], 
            linewidth = 2,
            dashes=dashes_choices[i],
            # linestyle = linestyles[i],
            label=
            '({:.2f}, {:.2f})'.format(input_CD[ind-1,0], input_CD[ind-1, 1])
            )
    axs[0].set_xlabel('log(particle arrivial time)')
    axs[0].set_ylabel('TCDF')   
    axs[0].xaxis.set_major_locator(mticker.LinearLocator(6))
    axs[0].xaxis.set_major_formatter('{x:.00f}') 
    axs[1].set_ylabel('log(particle arrivial time)')
    axs[1].set_xlabel('TCDF')      
    axs[1].yaxis.set_major_locator(mticker.LinearLocator(6))
    axs[1].yaxis.set_major_formatter('{x:.00f}')     
    # axs[0].legend(title='(C, D)=')
    axs[1].legend(title='(C, D)=')
    fig.savefig('uncorr6simu.pdf')
    
                
