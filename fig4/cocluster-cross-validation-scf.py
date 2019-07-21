# coding: utf-8

import sys
sys.path.insert(0, '/cndd/fangming/CEMBA/snmcseq_dev')
sys.path.insert(0, '/cndd/fangming/CEMBA/scripts/ClusterCrossValidation')
import importlib


from __init__ import *
from __init__jupyterlab import *
from scipy import sparse
import collections
import itertools
import re
import fbpca
from sklearn.model_selection import KFold
import pickle

import snmcseq_utils
import CEMBA_run_tsne
import CEMBA_clst_utils
import SCF_utils

from cluster_cv_utils import *

# # Configs  

DATA_DIR = '/cndd/fangming/CEMBA/data/MOp_all/organized_neurons_v6'
# fixed dataset configs
sys.path.insert(0, DATA_DIR)
from __init__datasets import *

meta_f = os.path.join(DATA_DIR, '{0}_metadata.tsv')
hvftrs_f = os.path.join(DATA_DIR, '{0}_hvfeatures.{1}')
hvftrs_gene = os.path.join(DATA_DIR, '{0}_hvfeatures.gene')
hvftrs_cell = os.path.join(DATA_DIR, '{0}_hvfeatures.cell')

mods_selected = [
    'snmcseq_gene',
    'snatac_gene',
    'smarter_cells',
    'smarter_nuclei',
    '10x_cells', 
    '10x_nuclei', 
    '10x_cells_v3',
#     '10x_nuclei_v3',
    '10x_nuclei_v3_Macosko',
    ]

features_selected = ['10x_cells']
# check features
for features_modality in features_selected:
    assert (features_modality in mods_selected)

# within modality
ps = {'mc': 0.9,
      'atac': 0.1,
      'rna': 0.7,
     }
drop_npcs = {'mc': 0,
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

# umap
umap_neighbors = 60
min_dist=min_dist = 0.5


# ## Read in data 

log = snmcseq_utils.create_logger()
logging.info('*')

# gene chrom lookup
chroms = np.arange(1, 20, 1).tolist() + ['X']
chroms = ['chr'+str(chrom) for chrom in chroms]

f = PATH_GENEBODY_ANNOTATION
df_genes = pd.read_csv(f, sep="\t")
gene_chrom_lookup = (df_genes[df_genes['chr'].isin(chroms)]
                            .groupby('gene_name').first()['chr']
                            .replace('chrX', 'chr20')
                            .apply(lambda x: int(x[3:]))
                   ) # 1:20

metas = collections.OrderedDict()
for mod in mods_selected:
    metas[mod] = pd.read_csv(meta_f.format(mod), sep="\t").reset_index().set_index(settings[mod].cell_col)
    print(mod, metas[mod].shape)

gxc_hvftrs = collections.OrderedDict()
for mod in mods_selected:
    print(mod)
    ti = time.time()
    
    if settings[mod].mod_category == 'mc':
        f_mat = hvftrs_f.format(mod, 'tsv')
        gxc_hvftrs[mod] = pd.read_csv(f_mat, sep='\t', header=0, index_col=0) 
        gxc_hvftrs[mod].index = SCF_utils.standardize_gene_name(gxc_hvftrs[mod].index)  # standardize gene name 
        print(gxc_hvftrs[mod].shape, time.time()-ti)
        assert np.all(gxc_hvftrs[mod].columns.values == metas[mod].index.values) # make sure cell name is in the sanme order as metas (important if save knn mat)
        continue
        
    f_mat = hvftrs_f.format(mod, 'npz')
    f_gene = hvftrs_gene.format(mod)
    f_cell = hvftrs_cell.format(mod)
    _gxc_tmp = snmcseq_utils.load_gc_matrix(f_gene, f_cell, f_mat)
    _gene = _gxc_tmp.gene
    _cell = _gxc_tmp.cell
    _mat = _gxc_tmp.data

    _gene = SCF_utils.standardize_gene_name(_gene)  # standardize gene name  
    
#     ## remove duplicated genes (for now)
#     u, c = np.unique(_gene, return_counts=True)
#     dup = u[c > 1]
#     uniq_bool = np.array([False if gene in dup else True for gene in _gene])
#     _gene_selected = _gene[uniq_bool]
#     _gene_selected_idx = np.arange(len(_gene))[uniq_bool]
#     _gene = _gene_selected
#     _mat = _mat.tocsr()[_gene_selected_idx, :]
#     ## remove duplicated genes complete
    
    gxc_hvftrs[mod] = GC_matrix(_gene, _cell, _mat)
    assert np.all(gxc_hvftrs[mod].cell == metas[mod].index.values) # make sure cell name is in the sanme order as metas (important if save knn mat)
    print(gxc_hvftrs[mod].data.shape, time.time()-ti)
    

# # subsample cells
# p = 1
# if p < 1:
#     metas_sub, gxc_hvftrs_sub = subsampling(mods_selected, metas, gxc_hvftrs, p)
# else:
#     metas_sub = metas
#     gxc_hvftrs_sub = gxc_hvftrs

resolutions = [0.1, 0.2, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 20, 30, 40, 60, 80, 100]
ns = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

for n in ns:
    name = 'mop_cv_scf_8mods_n{}_190719'.format(n)
    outdir = '/cndd/fangming/CEMBA/data/MOp_all/results'
    output_results = outdir + '/cross_validation_results_{}.pkl'.format(name)
    output_pcX_all = outdir + '/pcX_all_{}.npy'.format(name)
    output_cells_all = outdir + '/cells_all_{}.npy'.format(name)
    output_clst_and_umap = outdir + '/intg_summary_{}.tsv'.format(name)
    output_figures = outdir + '/figures/{}_{{}}.{{}}'.format(name)

    metas_sub, gxc_hvftrs_sub = subsampling(mods_selected, settings, metas, gxc_hvftrs, n=n)
    (
     res_nclsts, res, 
    ) = nfoldcv_scf_random_features_split(gxc_hvftrs_sub, resolutions, gene_chrom_lookup,
                                          mods_selected, metas_sub, 
                                          features_selected, settings, 
                                          ps, drop_npcs,
                                          cross_mod_distance_measure, knn, relaxation, n_cca,
                                          npc,
                                          output_pcX_all, output_cells_all, output_clst_and_umap,
                                          k=30, 
                                          reduce_dim=0,
                                          nfolds=5, n_repeats=5, n_splits=5, split_frac=0.5)


    # Saving the objects:
    with open(output_results, 'wb') as f: 
        pickle.dump((
                     res_nclsts, 
                     res,
                    ), f)

    # Getting back the objects:
    with open(output_results, 'rb') as f: 
        (res_nclsts, 
         res,
        ) = pickle.load(f)

    res_nclsts_summary = res_nclsts.groupby('resolution').agg({'nclsts': ['mean', 'std']})
    res_summary = res.groupby(['resolution', 'mod']).agg({'mse': ['mean', 'std'],
                                                          'mse_t': ['mean', 'std'],
                                                          })
    output = output_figures.format('cluster_cv', 'pdf')

    n_folds = 5
    n = len(mods_selected)
    nx = 3
    ny = int((n+nx-1)/nx)
    scale = 1
    fig, axs = plt.subplots(ny, nx, figsize=(5*nx*scale,4*ny*scale), sharex=True, sharey=False)
    axs = axs.flatten()
    for i, (mod, ax) in enumerate(zip(mods_selected, axs)):
        x = res_nclsts_summary['nclsts']['mean'].values
        ys = res_summary.xs(mod, level='mod')
        base_level = np.min(ys['mse']['mean'].values)
        y, y_err = ys['mse']['mean'].values/base_level, (ys['mse']['std'].values/np.sqrt(n_folds))/base_level, 
        yt, yt_err = ys['mse_t']['mean'].values/base_level, (ys['mse_t']['std'].values/np.sqrt(n_folds))/base_level, 
        
        if i % nx == 0:
            ylabel = 'MSE +/- SEM\n(normalized)'
        else:
            ylabel = ''
        xlabel = ''
        min_x_se, min_x, min_y = plot_bi_cv_subfig(ax, x, 
                                                  y, y_err,
                                                  yt, yt_err,
                                                  settings[mod].color, mod, 
                                                  xlabel=xlabel,
                                                  ylabel=ylabel
                                                 )
        ax.yaxis.set_major_locator(mtick.MaxNLocator(4))

    fig.subplots_adjust(wspace=0.3, bottom=0.15)
    fig.text(0.5, 0, 'Number of clusters', ha='center', fontsize=15)
    fig.savefig(output, bbox_inches='tight')
    plt.show()
    plt.close()

