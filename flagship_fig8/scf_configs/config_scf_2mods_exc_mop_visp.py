#!/usr/bin/env python3
"""An example configuration file
"""
import sys
import os

# # Configs  
name = 'mop_visp_2mods_exc' 
outdir = '../results_replicate'
output_pcX_all = outdir + '/pcX_all_{}.npy'.format(name)
output_cells_all = outdir + '/cells_all_{}.npy'.format(name)
output_imputed_data_format = outdir + '/imputed_data_{}_{{}}.npy'.format(name)
output_clst_and_umap = outdir + '/intg_summary_{}.tsv'.format(name)
output_figures = outdir + '/figures/{}_{{}}.{{}}'.format(name)


DATA_DIR = '../data_2mods_exc_mop_visp'
# fixed dataset configs
sys.path.insert(0, DATA_DIR)
from __init__datasets import *

meta_f = os.path.join(DATA_DIR, '{0}_metadata.tsv')
hvftrs_f = os.path.join(DATA_DIR, '{0}_hvfeatures.{1}')
hvftrs_gene = os.path.join(DATA_DIR, '{0}_hvfeatures.gene')
hvftrs_cell = os.path.join(DATA_DIR, '{0}_hvfeatures.cell')

mods_selected = [
    'smarter_cells',
    'smarter-cells-v1',
    ]

features_selected = ['smarter-cells-v1']

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
knn = 10 
relaxation = 3 
n_cca = 30

# PCA
npc = 50

# clustering
k = 30
resolutions = [0.1, 1, 2, 4]
# umap
umap_neighbors = 30
min_dist = 0.5
