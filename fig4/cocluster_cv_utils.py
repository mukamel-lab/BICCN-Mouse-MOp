
# coding: utf-8

# In[29]:


import sys
sys.path.insert(0, '/cndd/fangming/CEMBA/snmcseq_dev')
import importlib


import logging
from __init__ import *
from __init__jupyterlab import *
from scipy import sparse
import collections
import itertools
import re
import fbpca

import snmcseq_utils
importlib.reload(snmcseq_utils)
import CEMBA_run_tsne
import CEMBA_clst_utils
import SCF_utils
importlib.reload(SCF_utils)

import pickle

# import seaborn as sns 
# import matplotlib.pyplot as plt
logger = snmcseq_utils.create_logger()

# In[2]:
def nfoldcv_scf(gxc_hvftrs_sub_g0, gxc_hvftrs_sub_g1, resolutions, k, 
                metas_sub, mods_selected, features_selected, settings,   
                ps, drop_npcs,
                cross_mod_distance_measure, knn, relaxation, n_cca,
                npc,
                output_pcX_all, output_cells_all, output_clst_and_umap,
                reduce_dim=0,
                nfolds=5, n_repeats=10):
    """
    """
    from sklearn.model_selection import KFold
    
#     if os.path.isfile(output_clst_and_umap):
#         df_clsts = pd.read_csv(output_clst_and_umap, sep="\t", index_col=0)
        
#     else:
    # cluster on g0 with different resolutions
    pcX_all, cells_all = SCF_utils.core_scf_routine(mods_selected, features_selected, settings, 
                                                    metas_sub, gxc_hvftrs_sub_g0, 
                                                    ps, drop_npcs,
                                                    cross_mod_distance_measure, knn, relaxation, n_cca,
                                                    npc,
                                                    output_pcX_all, output_cells_all,
                                                    )
    umap_neighbors = 60 # doesn't matter
    min_dist = 0.7 # doesn't matter
    df_clsts = SCF_utils.clustering_umap_routine(pcX_all, cells_all, mods_selected, metas_sub,
                                                 resolutions, k, 
                                                 umap_neighbors, min_dist, # doesn't matter
                                                 output_clst_and_umap,
                                                 cluster_only=True,
                                                 )
    
    # train and predict on g1
    res_nclsts = []
    res_mse_mean = {mod: [] for mod in mods_selected} 
    res_mse_se = {mod: [] for mod in mods_selected} 
    res_mse_t_mean = {mod: [] for mod in mods_selected} 
    res_mse_t_se = {mod: [] for mod in mods_selected}
    kl = KFold(n_splits=nfolds)
    
    # test different resolution
    for resolution in resolutions:
        logging.info(resolution)
        res_mse = {}
        res_mse_t = {} 

        cluster_col = 'cluster_joint_r{}'.format(resolution)
        df_clst = df_clsts[[cluster_col]].rename(columns={cluster_col: 'cluster'})
        nclsts = len(df_clst['cluster'].unique())
        res_nclsts.append(nclsts) # record number of clusters
        cells_clst = df_clst['cluster'] # cell -> cluster label look up series
        
        # do it in every modality
        for mod in mods_selected:
            logging.info(mod)
            # set up
            res_mse[mod] = []
            res_mse_t[mod] = []
            
            metadata = metas_sub[mod].copy()
            metadata['cluster_cv'] = df_clst.loc[metadata.index, 'cluster'] 
            gxc_hvftr = gxc_hvftrs_sub_g1[mod]
            
            if settings[mod].mod_category == 'mc':
                assert np.all(metadata.index.values == gxc_hvftr.columns.values)
                features_y = gxc_hvftr.T.values
                if reduce_dim:
                    U, s, Vt = fbpca.pca(features_y, k=reduce_dim)
                    features_y = U.dot(np.diag(s))
            else:
                assert np.all(metadata.index.values == gxc_hvftr.cell)
                features_y = pd.DataFrame(gxc_hvftr.data.T.todense(), 
                                          index=gxc_hvftr.cell, 
                                          columns=gxc_hvftr.gene).values
                if reduce_dim:
                    U, s, Vt = fbpca.pca(features_y, k=reduce_dim)
                    features_y = U.dot(np.diag(s))
                
            ncells = len(metadata)
            
            for i_repeat in range(n_repeats):
                if i_repeat % 5 == 0:
                    logging.info(i_repeat)

                # shuffle data
                cells_shuffled_idx = np.random.permutation(np.arange(ncells))
                metadata = metadata.iloc[cells_shuffled_idx, :] 
                metadata['cell_idx'] = np.arange(ncells)
                features_y = features_y[cells_shuffled_idx, :]
                
                # split training and test 
                for train_idx, test_idx in kl.split(np.arange(ncells)):
                    ti = time.time()
#                     print(0, time.time()-ti)
                    # compute cluster centroids for training cells 
                    metadata_train = metadata.iloc[train_idx]
                    clsts_in_train = np.unique(metadata_train['cluster_cv'].values)
                    clsts_not_in_train = np.unique(metadata['cluster_cv'].values).tolist()
                    y_centroids = np.zeros((len(clsts_in_train), features_y.shape[1]))
                    cluster_to_idx_lookup = {}
                    for count_idx, (clst, df_sub) in enumerate(metadata_train.groupby('cluster_cv')):
                        cells_sub_idx = df_sub['cell_idx'].values
                        y_centroids[count_idx, :] = features_y[cells_sub_idx, :].mean(axis=0)
                        cluster_to_idx_lookup[clst] = count_idx
                        clsts_not_in_train.remove(clst)
                    for clst in clsts_not_in_train:
                        cluster_to_idx_lookup[clst] = -1
                    
                    # compute MSE for test cells
                    cells_j = metadata.index.values[test_idx]
                    clsts_i = cells_clst[cells_j]
                    clsts_i_idx = np.array([cluster_to_idx_lookup[clst] for clst in clsts_i])
                    cond = (clsts_i_idx != -1)  # test if clsts_i in clsts_in_train
                    test_idx, cells_j, clsts_i, clsts_i_idx = test_idx[cond], cells_j[cond], clsts_i[cond], clsts_i_idx[cond]
                    diff = features_y[test_idx, :] - y_centroids[clsts_i_idx, :]
                    mse = (diff**2).sum(axis=1).mean()
                    res_mse[mod].append(mse)
                    
                    # compute MSE for training cells 
                    cells_j = metadata.index.values[train_idx]
                    clsts_i = cells_clst[cells_j]
                    clsts_i_idx = np.array([cluster_to_idx_lookup[clst] for clst in clsts_i])
                    diff = features_y[train_idx, :] - y_centroids[clsts_i_idx, :]
                    mse = (diff*diff).sum(axis=1).mean()
                    res_mse_t[mod].append(mse)
                    
                # end of n-fold training test for 
                # each collect 1 data point
#                 break
                
            # end of n-repeats for
            # summarize n-repeats into stats
#             break
            res_mse[mod] = np.array(res_mse[mod])
            res_mse_mean[mod].append(res_mse[mod].mean())
            res_mse_se[mod].append(1.96*res_mse[mod].std()/np.sqrt(nfolds))

            res_mse_t[mod] = np.array(res_mse_t[mod])
            res_mse_t_mean[mod].append(res_mse_t[mod].mean())
            res_mse_t_se[mod].append(1.96*res_mse_t[mod].std()/np.sqrt(nfolds))
            print('')
#         break
        # end of n-modality for 
        
#     break
    # end of resolution for
    
    res_nclsts = np.array(res_nclsts)
    for mod in mods_selected:
        res_mse_mean[mod] = np.array(res_mse_mean[mod])
        res_mse_se[mod] = np.array(res_mse_se[mod])
        res_mse_t_mean[mod] = np.array(res_mse_t_mean[mod])
        res_mse_t_se[mod] = np.array(res_mse_t_se[mod])

    return ( 
         res_nclsts,
         res_mse_mean, res_mse_se, 
         res_mse_t_mean, res_mse_t_se,
        )
    

# In[22]:


# plotting utils

def plot_errorbar_ax(ax, x, y, yerr, color='C0', label=''):
    """Plot a line with errorbar 
    """
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
    
    ax.plot(x, y, '-o', 
           markersize=5,
           color=color,
           label=label,
           )
    ax.fill_between(x, y-yerr, y+yerr, 
                    color=color,
                    alpha=0.3,
                    zorder=0,
                   )
    return

def plot_errorbar_fancymin_ax(ax, x, y, yerr, color='C0', label=''):
    """Plot a line with errorbar + min position and min-se position
    """
    from scipy import optimize
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
    plot_errorbar_ax(ax, x, y, yerr, color=color, label=label)
    
    # get minimum and plot
    min_arg = np.argmin(y)
    min_x = x[min_arg]
    min_y = y[min_arg]
    ax.plot(min_x, min_y, '^',
               markersize=12,
               color=color,
               )
    
    # get minimum + se and plot
    epsilon = 0
    f = lambda _x: np.interp(_x, x[:min_arg], (y-yerr)[:min_arg]) - (min_y + epsilon)
    try:
        res_root = optimize.root_scalar(f, bracket=(x[0], min_x))
        min_x_se = int(res_root.root+0.5)
    except: 
        if np.all(f(x[:min_arg])<0):
            min_x_se = x[0]
        elif np.all(f(x[:min_arg])>0):
            min_x_se = x[min_arg]
        else:
            raise ValueError("Dont understand f: {}".format(f(x[:min_arg])))
    ax.plot(min_x_se, min_y, 's', 
               markersize=10,
               color=color,
           )
        
    return int(min_x_se), int(min_x), min_y
    
def plot_bi_cv_ax(ax, x, y, yerr, color='C0', mod="", ylabel="MSE +/- SEM Normalized"):
    """
    """
    min_x_se, min_x, min_y = plot_errorbar_fancymin_ax(ax, x, y, yerr, color=color,)
    ax.set_title("{}: {} - {}".format(mod, min_x_se, min_x))
    ax.set_ylabel(ylabel)
    return

def plot_bi_cv_subfig(ax, x1, y1, yerr1, y1_tr, yerr1_tr, color1, mod1,
                    xlabel='Number of clusters',
                    ylabel='MSE +/- SEM Normalized',
                   ):
    from matplotlib.ticker import ScalarFormatter
    
    plot_errorbar_ax(ax, x1, y1_tr, yerr1_tr, color='black', label='Training error')
    plot_bi_cv_ax(ax, x1, y1, yerr1, color=color1, mod=mod1, ylabel=ylabel)
        
    ax.set_xscale('log', basex=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(xlabel)
    
    return
