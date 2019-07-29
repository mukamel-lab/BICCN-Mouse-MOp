
# coding: utf-8

# In[30]:


import importlib
import sys
sys.path.insert(0, '/cndd/fangming/CEMBA/snmcseq_dev')

from __init__ import *
from __init__jupyterlab import *
from scipy import sparse
import collections
# import itertools
# import re
# import fbpca
# import pickle
# import scipy.cluster.hierarchy as sch
# from sklearn.metrics import euclidean_distances

import snmcseq_utils
importlib.reload(snmcseq_utils)
# import CEMBA_run_tsne
# import CEMBA_clst_utils
# import SCF_utils
# importlib.reload(SCF_utils)
import CEMBA_preproc_utils
importlib.reload(CEMBA_preproc_utils)



# In[6]:


name = 'mop_8mods_datav8_190723'
outdir = '/cndd/fangming/CEMBA/data/MOp_all/results'
output_pcX_all = outdir + '/pcX_all_{}.npy'.format(name)
output_cells_all = outdir + '/cells_all_{}.npy'.format(name)
output_imputed_data_format = outdir + '/imputed_data_{}_{{}}.npy'.format(name)
output_clst_and_umap = outdir + '/intg_summary_{}.tsv'.format(name)
output_figures = outdir + '/figures/{}_{{}}.{{}}'.format(name)
output_cluster_centroids = outdir + '/centroids_{}.pkl'.format(name)


# In[7]:


DATA_DIR = '/cndd/fangming/CEMBA/data/MOp_all/organized_neurons_v8'

# fixed dataset configs
sys.path.insert(0, DATA_DIR)
import __init__datasets
importlib.reload(__init__datasets)
from __init__datasets import *

meta_f = '{0}/{1}_metadata.tsv'
raw_f = '{0}/{1}_{2}raw.{3}'
hvftr_f = '{0}/{1}_hvfeatures.{2}'


# In[8]:


mods_selected = [
    'snmcseq_gene',
    'snatac_gene',
    'smarter_cells',
    'smarter_nuclei',
    '10x_cells', 
    '10x_cells_v3',
    '10x_nuclei_v3',
    '10x_nuclei_v3_Macosko',
    ]


# In[9]:


log = snmcseq_utils.create_logger()
logging.info('*')


# In[10]:


DST_ROOT_DIR = '/cndd/fangming/CEMBA/data/MOp_all/organized_neurons_subtypes_v8'


# In[11]:


metas = collections.OrderedDict()
for mod in mods_selected:
    metas[mod] = pd.read_csv(meta_f.format(DATA_DIR, mod), sep="\t").reset_index().set_index(settings[mod].cell_col)
    print(mod, metas[mod].shape)


# In[12]:


gxc_raws = collections.OrderedDict()
for mod in mods_selected:
    logging.info("Read data {}...".format(mod))
    if settings[mod].mod_category == 'mc':
        f_gene = raw_f.format(DATA_DIR, mod, '', 'gene')
        f_cell = raw_f.format(DATA_DIR, mod, '', 'cell')
        f_data_c = raw_f.format(DATA_DIR, mod, 'CH_', 'npz')
        f_data_mc = raw_f.format(DATA_DIR, mod, 'mCH_', 'npz')
        gxc_raws[mod] = snmcseq_utils.load_gc_matrix_methylation(f_gene, f_cell, f_data_mc, f_data_c)
        
    else:
        f_gene = raw_f.format(DATA_DIR, mod, '', 'gene')
        f_cell = raw_f.format(DATA_DIR, mod, '', 'cell')
        f_data = raw_f.format(DATA_DIR, mod, '', 'npz')
        gxc_raws[mod] = snmcseq_utils.load_gc_matrix(f_gene, f_cell, f_data)
        


# In[13]:


f = output_clst_and_umap
first_round_cluster_col = 'cluster_joint_r0.1'
df_info = pd.read_csv(f, sep="\t", index_col='sample')[[first_round_cluster_col, 'modality']] 
print(df_info.shape)
df_info.head()


# In[20]:


normalization_options = {
    'smarter_nuclei': 'TPM',
    'smarter_cells': 'TPM',
    'snatac_gene': 'TPM',
    '10x_nuclei_v3_Macosko': 'CPM',
    '10x_cells_v3': 'CPM',
    '10x_nuclei_v3': 'CPM',
    'snmcseq_gene': 'MC',
    '10x_cells': 'CPM',
}

gene_annot_file = PATH_GENEBODY_ANNOTATION
gene_annot = pd.read_csv(gene_annot_file, sep="\t")
gene_annot_v2 = gene_annot.groupby('gene_name').first()
print(gene_annot_v2.shape)
gene_lengths_base = (gene_annot_v2['end'] - gene_annot_v2['start'])
print(gene_lengths_base.head())


# In[31]:



logging.info("Prep data...")
for (mod, clst), df_sub in df_info.groupby(['modality', first_round_cluster_col]):
    if mod in mods_selected:
        print(mod, clst)
        ti = time.time()
        normalization_option = normalization_options[mod]

        _cells = df_sub.index.values
        dst_dir = os.path.join(DST_ROOT_DIR, str(clst))
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)

        # meta and save meta
        meta_f_dst = meta_f.format(dst_dir, mod)
        meta = metas[mod].loc[_cells]
        meta.to_csv(meta_f_dst, sep="\t", header=True, index=True)

        if settings[mod].mod_category == 'mc':
            # split raw data
            _cell_idx = snmcseq_utils.get_index_from_array(gxc_raws[mod].cell, _cells)
            gxc_raw = GC_matrix(
                gxc_raws[mod].gene,
                gxc_raws[mod].cell[_cell_idx],
                {'mc': gxc_raws[mod].data['mc'].tocsc()[:, _cell_idx],
                 'c': gxc_raws[mod].data['c'].tocsc()[:, _cell_idx],
                },
            )
            # check meta cells agree with gxc cells
            assert np.all(meta.index.values == gxc_raw.cell)
            # check genes are uniq 
            assert len(gxc_raw.gene) == len(np.unique(gxc_raw.gene)) 
            
            # get hvftrs
            gxc_hvftr = CEMBA_preproc_utils.preproc_methylation(gxc_raw, meta,
                                                                global_value_col=settings[mod].global_mean, 
                                                                base_call_cutoff=20, 
                                                                sufficient_coverage_fraction=0.95,
                                                                hv_percentile=30,
                                                                n_qcut=10,
                                                               )

            # save data
            print(mod, "Saving to files {}".format(time.time()-ti))
            f_data = hvftr_f.format(dst_dir, mod, 'tsv') 
            gxc_hvftr.to_csv(f_data, sep="\t", index=True, header=True)
            logging.info("{} {} Total time used: {}".format(mod, clst, time.time()-ti))
            
        else:
            # split raw data
            _cell_idx = snmcseq_utils.get_index_from_array(gxc_raws[mod].cell, _cells)
            gxc_raw = GC_matrix(
                gxc_raws[mod].gene,
                gxc_raws[mod].cell[_cell_idx],
                gxc_raws[mod].data.tocsc()[:, _cell_idx],
            )
            # check meta cells agree with gxc cells
            assert np.all(meta.index.values == gxc_raw.cell)
            # check genes are uniq 
            assert len(gxc_raw.gene) == len(np.unique(gxc_raw.gene)) 

            # get hvftrs
            print(mod, "Preproc and get highly variable genes {}".format(time.time()-ti))
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
                
            # save data
            print(mod, "Saving to files {}".format(time.time()-ti))
            f_gene = hvftr_f.format(dst_dir, mod, 'gene') 
            f_cell = hvftr_f.format(dst_dir, mod, 'cell') 
            f_data = hvftr_f.format(dst_dir, mod, 'npz') 
            snmcseq_utils.save_gc_matrix(gxc_hvftr, f_gene, f_cell, f_data)
            logging.info("{} {} Total time used: {}".format(mod, clst, time.time()-ti))

    
    

