#!/usr/bin/env python3
"""An example configuration file
"""
import sys
sys.path.insert(0, '/cndd/fangming/CEMBA/snmcseq_dev')
import os
import snmcseq_utils

# # Configs  
name = 'mop_8mods_0915_k30_sub3-2-1'
outdir = '/cndd/fangming/CEMBA/data/MOp_all/results'
output_pcX_all = outdir + '/pcX_all_{}.npy'.format(name)
output_cells_all = outdir + '/cells_all_{}.npy'.format(name)
output_imputed_data_format = outdir + '/imputed_data_{}_{{}}.npy'.format(name)
output_clst_and_umap = outdir + '/intg_summary_{}.tsv'.format(name)
output_figures = outdir + '/figures/{}_{{}}.{{}}'.format(name)
output_cluster_centroids = outdir + '/centroids_{}.pkl'.format(name)


DATA_DIR = '/cndd/fangming/CEMBA/data/MOp_all/data_freeze_neurons_subtypes_8mods_round4/sub3-2-1'
# fixed dataset configs
sys.path.insert(0, DATA_DIR)
from __init__datasets import *

meta_f = os.path.join(DATA_DIR, '{0}_metadata.tsv')
hvftrs_f = os.path.join(DATA_DIR, '{0}_hvfeatures.{1}')
hvftrs_gene = os.path.join(DATA_DIR, '{0}_hvfeatures.gene')
hvftrs_cell = os.path.join(DATA_DIR, '{0}_hvfeatures.cell')

# mods_selected = [
#     'snmcseq_gene',
#     'snatac_gene',
#     'smarter_cells',
#     'smarter_nuclei',
#     '10x_cells_v2', 
#     '10x_cells_v3',
#     '10x_nuclei_v3',
#     '10x_nuclei_v3_macosko',
#     ]
mods_selected = snmcseq_utils.import_single_textcol(os.path.join(DATA_DIR, 'datasets.txt'))
print(mods_selected)

features_selected = ['10x_cells_v2']
# check features
for features_modality in features_selected:
    assert (features_modality in mods_selected)

# within modality
ps = {'mc': 0.9,
      'atac': 0.1,
      'rna': 0.7,
     }
drop_npcs = {
      'mc': 0,
      'atac': 0,
      'rna': 0,
     }

# across modality
cross_mod_distance_measure = 'correlation' # cca
knn = 20 
relaxation = 3
n_cca = 30

# PCA
npc = 50

# clustering
k = 30
resolutions = [0.1, 0.2, 0.5, 1,]
# umap
umap_neighbors = 30
min_dist = 0.5
