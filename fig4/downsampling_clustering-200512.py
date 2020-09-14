
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '/cndd/fangming/CEMBA/snmcseq_dev')
sys.path.insert(0, '/cndd/fangming/CEMBA/scripts/ClusterCrossValidation')
import importlib

from __init__ import *
from __init__jupyterlab import *


import collections
from collections import deque
from scipy import stats
from scipy import optimize 
from scipy.optimize import curve_fit

import queue
# import tables
from scipy import sparse
from sklearn.model_selection import KFold
# from sklearn.decomposition import PCA
# from sklearn.neighbors import NearestNeighbors
# from sklearn.utils.sparsefuncs import mean_variance_axis
import fbpca
from statsmodels.stats.multitest import multipletests
import datetime


import snmcseq_utils
importlib.reload(snmcseq_utils)
import CEMBA_clst_utils
importlib.reload(CEMBA_clst_utils)
import cluster_cv_utils
import CEMBA_preproc_utils

# from CEMBA_run_tsne import run_tsne
# from CEMBA_run_tsne import run_tsne_v2


# ## Basic settings 
# - use ```mods``` and ```settings[mod]``` to access modality specific information

# In[2]:


mods_selected = [
#     'snmcseq_gene',
#     'snatac_gene',
    # 'smarter_cells',
    # 'smarter_nuclei',
    '10x_cells_v2', 
    '10x_cells_v3',
    '10x_nuclei_v3',
    '10x_nuclei_v3_macosko',
    ]

SRC_DIR = '/cndd/fangming/CEMBA/data/MOp_all/data_freeze_neurons'
DST_DIR = '/cndd2/fangming/projects/miniatlas'


# In[3]:


# # gene id (abbr) as index
gene_annot_file = PATH_GENEBODY_ANNOTATION
gene_annot = pd.read_csv(gene_annot_file, sep="\t")
gene_annot['gene_id_abbr'] = gene_annot['gene_id'].apply(lambda x: x.split('.')[0])
gene_annot = gene_annot.set_index('gene_id_abbr')

gene_lengths_base = (gene_annot['end'] - gene_annot['start'])
print(gene_lengths_base.head())


# In[4]:


DATA_DIR = '/cndd/fangming/CEMBA/data/MOp_all/data_freeze_neurons'

# fixed dataset configs
sys.path.insert(0, DATA_DIR)
import __init__datasets
importlib.reload(__init__datasets)
from __init__datasets import *


f_meta_format = '{0}/{1}_metadata.tsv'
f_data_format = '{0}/{1}_{2}raw.{3}'
f_hvftr_format = '{0}/{1}_hvfeatures.{2}'
f_ftr_format = '{0}/{1}_features.{2}'


# In[5]:


# downsample_cells = [1000, 2000, 4000, 5911]
# downsample_reads = np.array([1e4, 3e4, 5e4, 1e5, 3e5, 5e5, 0.98*1e6])

# downsample_cells = [2000, 4000, 5911]
# downsample_reads = np.array([1e4, 3e4, 5e4, 1e5, 3e5, 5e5,])

# 10x
downsample_cells = [2000, 4000, 6000, 10000, 20000]
# downsample_reads = np.array([1e4, 2e4, 3e4, 4e4, 5e4,])
downsample_reads = np.array([1e3, 2e3, 5e3,])


# In[6]:


normalization_options = {
    'smarter_nuclei': 'TPM',
    'smarter_cells': 'TPM',
    '10x_nuclei_v3_macosko': 'CPM',
    '10x_cells_v3': 'CPM',
    '10x_nuclei_v3': 'CPM',
    '10x_cells_v2': 'CPM',
}


# In[7]:


# read in matrix

metas = collections.OrderedDict()
gxc_raws = collections.OrderedDict()
num_reads_all = collections.OrderedDict()


for mod in mods_selected:
    ti = time.time()
    print(mod)
    
    ## read data
    # read metadata
    normalization_option = normalization_options[mod]
    f_meta = f_meta_format.format(SRC_DIR, mod) ##
    meta = pd.read_csv(f_meta, sep="\t", index_col=0)
    metas[mod] = meta
        
    f_data = f_data_format.format(SRC_DIR, mod, '', 'npz') 
    f_data_gene = f_data_format.format(SRC_DIR, mod, '', 'gene') 
    f_data_cell = f_data_format.format(SRC_DIR, mod, '', 'cell') 
    
    # read counts matrix 
    print(mod, "Reading in files {}".format(time.time()-ti))
    gxc_raw = snmcseq_utils.load_gc_matrix(f_data_gene, f_data_cell, f_data) # checked dimensions in agreement internally
    gxc_raws[mod] = gxc_raw
    
    num_cells = len(meta)
    num_reads = gxc_raw.data.sum().sum()/num_cells
    num_reads_all[mod] = num_reads
    
    print(gxc_raw.data.shape, num_cells, num_reads)
    
    # check meta cells agree with gxc cells
    assert np.all(meta.index.values == gxc_raw.cell)
    # check genes are uniq 
    assert len(gxc_raw.gene) == len(np.unique(gxc_raw.gene)) 
    
    print(mod, "Total time used: {}".format(time.time()-ti))


# In[8]:


def downsamp_analysis(downsample_cell, downsample_read, mods_selected, settings, metas, gxc_raws, n_repeat_start=0): 
    """
    """
    summary = []
    
    for i_repeat in range(n_repeat): # repeat subsample
        i_repeat += n_repeat_start
        ## downsample cells
        n = downsample_cell
        metas_sub, gxc_raws_sub = cluster_cv_utils.subsampling(mods_selected, settings, metas, gxc_raws, p=0, n=n)

        for mod in mods_selected:
            ti = time.time()
            meta = metas_sub[mod]
            gxc_raw = gxc_raws_sub[mod]

            ## downsample reads
            num_read = gxc_raw.data.sum().sum()/downsample_cell
            p = downsample_read/num_read
            if p < 1:
                gxc_raw.data.data = np.random.binomial(gxc_raw.data.data, p)
                downsample_read_update = downsample_read
            else:
                downsample_read_update = num_read


            # get hvftrs
            if normalization_option == 'CPM':
                gxc_hvftr = CEMBA_preproc_utils.preproc_rna_cpm_based(
                                                 gxc_raw, 
                                                 sufficient_cell_coverage=0.01, 
                                                 hv_percentile=30, hv_ncut=10)
            elif normalization_option == 'TPM':
                gene_lengths = gene_lengths_base.reindex(gxc_raw.gene)
                gxc_hvftr = CEMBA_preproc_utils.preproc_rna_tpm_based(
                                                 gxc_raw, gene_lengths, impute_gene_lengths=True, 
                                                 sufficient_cell_coverage=0.01, 
                                                 hv_percentile=30, hv_ncut=10)

            ## cluster 
            # cell-by-gene matrix
            X = pd.DataFrame(gxc_hvftr.data.T.todense(), 
                            index=gxc_hvftr.cell,
                            columns=gxc_hvftr.gene,
                            ) 

            # PCA
            U, s, Vt = fbpca.pca(X.values, npc)
            pcX = U.dot(np.diag(s)) 
            cell_list = X.index.values

            # Clustering 
            for r in rs:  # stringency parameters
                df_clst = CEMBA_clst_utils.clustering_routine(pcX, cell_list, k, 
                                                             resolution=r,
                                                             seed=1, verbose=False,
                                                             metric='euclidean', option='plain', 
                                                             n_trees=10, search_k=-1, num_starts=None)

                output = output_format.format(mod, downsample_cell, downsample_read, r, i_repeat) 
                df_clst.to_csv(output, sep='\t', na_rep='NA', index=True, header=True)
                nclst = len(df_clst['cluster'].unique())
                summary.append({
                    'ncell': downsample_cell,
                    'nread': downsample_read_update,
                    'mod': mod, 
                    'resolution': r,
                    'repeat': i_repeat,
                })

                print("Number of clusters: {}".format(nclst))
                print(mod, "Total time used: {}".format(time.time()-ti))
    return summary


# In[9]:



output_format = ('/cndd2/fangming/projects/miniatlas/'
                 'results/clst_neuron_downsamp_mod-{0}_ncell-{1}_nread-{2}_r-{3}_i-{4}_200512.tsv'
                )

n_repeat = 10 
n_repeat_start = 0
k = 30
npc = 50
# rs = [1, 6, ]
rs = [6,]


summary = []
        

for downsample_cell in downsample_cells:
    for downsample_read in downsample_reads:
        print("******* {} {}".format(downsample_cell, downsample_read))
        summary_singleround = downsamp_analysis(downsample_cell, downsample_read, 
                                                  mods_selected, settings, metas, gxc_raws, n_repeat_start=n_repeat_start)
        summary += summary_singleround 
        
summary = pd.DataFrame(summary) 
print(summary.shape)
output = '/cndd2/fangming/projects/miniatlas/results/summary_downsamp_{}.tsv'.format(datetime.datetime.now().date())
summary.to_csv(output, sep='\t', header=True, index=False)

summary.head()

