from cocluster_cv_utils import *

# # Configs  
# In[30]:

name = 'mop_cv_4mods_190710'
outdir = '/cndd/fangming/CEMBA/data/MOp_all/results'
output_results = outdir + '/cross_validation_results_{}.pkl'.format(name)
output_pcX_all = outdir + '/pcX_all_{}.npy'.format(name)
output_cells_all = outdir + '/cells_all_{}.npy'.format(name)
output_clst_and_umap = outdir + '/intg_summary_{}.tsv'.format(name)

output_figures = outdir + '/figures/{}_{{}}.{{}}'.format(name)

DATA_DIR = '/cndd/fangming/CEMBA/data/MOp_all/organized_neurons_v6'

# fixed dataset configs
sys.path.insert(0, DATA_DIR)
import __init__datasets
importlib.reload(__init__datasets)
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
    # '10x_cells', 
    # '10x_nuclei', 
    # '10x_cells_v3',
    # '10x_nuclei_v3',
    # '10x_nuclei_v3_Macosko',
    ]

features_selected = ['smarter_cells']
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
resolutions = [0.8, 1, 2, 4]
# umap
umap_neighbors = 60
min_dist=min_dist = 0.5


# ## Read in data 

# In[7]:


log = snmcseq_utils.create_logger()
logging.info('*')


# In[8]:


# gene chrome lookup
chroms = np.arange(1, 20, 1).tolist() + ['X']
chroms = ['chr'+str(chrom) for chrom in chroms]

f = PATH_GENEBODY_ANNOTATION
df_genes = pd.read_csv(f, sep="\t")
gene_set_lookup = (df_genes[df_genes['chr'].isin(chroms)]
                             .groupby('gene_name').first()['chr']
                             .replace('chrX', 'chr20')
                             .apply(lambda x: int(x[3:])%2)
                    )
print(gene_set_lookup.value_counts())
gene_set_lookup.head()


# In[9]:


metas = collections.OrderedDict()
for mod in mods_selected:
    metas[mod] = pd.read_csv(meta_f.format(mod), sep="\t").reset_index().set_index(settings[mod].cell_col)
    print(mod, metas[mod].shape)


# In[10]:


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
    
    
    gxc_hvftrs[mod] = GC_matrix(_gene, _cell, _mat)
    assert np.all(gxc_hvftrs[mod].cell == metas[mod].index.values) # make sure cell name is in the sanme order as metas (important if save knn mat)
    print(gxc_hvftrs[mod].data.shape, time.time()-ti)
    


# In[11]:


# subsample cells
p = 1

if p < 1:
    metas_sub = collections.OrderedDict()
    gxc_hvftrs_sub = collections.OrderedDict()
    for mod in mods_selected: 
        # subsample meta
        cells_included = metas[mod].index.values[np.random.rand(len(metas[mod]))<p]
        metas_sub[mod] = metas[mod].loc[cells_included]

        # subsample gxc_hvftrs
        if settings[mod].mod_category == 'mc':
            gxc_hvftrs_sub[mod] = gxc_hvftrs[mod][cells_included]
            print(mod, metas_sub[mod].shape, gxc_hvftrs_sub[mod].shape, time.time()-ti)
            continue

        cells_included_idx = snmcseq_utils.get_index_from_array(gxc_hvftrs[mod].cell, cells_included)
        gxc_hvftrs_sub[mod] = GC_matrix(
                                        gxc_hvftrs[mod].gene,
                                        cells_included,
                                        gxc_hvftrs[mod].data.tocsc()[:, cells_included_idx],
                                        )
        print(mod, metas_sub[mod].shape, gxc_hvftrs_sub[mod].data.shape, time.time()-ti)
else:
    metas_sub = metas
    gxc_hvftrs_sub = gxc_hvftrs


# In[12]:


# split features metas_sub, gxc_hvftrs_sub_g0, gxc_hvftrs_sub_g1

gxc_hvftrs_sub_g0 = collections.OrderedDict()
gxc_hvftrs_sub_g1 = collections.OrderedDict()
for mod in mods_selected: 
    # split gxc_hvftrs
    if settings[mod].mod_category == 'mc':
        _genes = gxc_hvftrs_sub[mod].index.values
        _lookup = gene_set_lookup.reindex(_genes).fillna(-1).values
        _genes_set0 = _genes[_lookup == 0]
        _genes_set1 = _genes[_lookup == 1]
        print(len(_genes_set0), len(_genes_set1))
        gxc_hvftrs_sub_g0[mod] = gxc_hvftrs_sub[mod].loc[_genes_set0]
        gxc_hvftrs_sub_g1[mod] = gxc_hvftrs_sub[mod].loc[_genes_set1]
        
        print(mod, gxc_hvftrs_sub_g0[mod].shape, gxc_hvftrs_sub_g1[mod].shape, time.time()-ti)
        continue
        
    _genes = gxc_hvftrs_sub[mod].gene
    _lookup = gene_set_lookup.reindex(_genes).fillna(-1).values
    _genes_set0 = _genes[_lookup == 0]
    _genes_set0_index = snmcseq_utils.get_index_from_array(_genes, _genes_set0)
    _genes_set1 = _genes[_lookup == 1]
    _genes_set1_index = snmcseq_utils.get_index_from_array(_genes, _genes_set1)
    print(len(_genes_set0), len(_genes_set1))
    gxc_hvftrs_sub_g0[mod] = GC_matrix(
                                _genes_set0,
                                gxc_hvftrs_sub[mod].cell,
                                gxc_hvftrs_sub[mod].data.tocsr()[_genes_set0_index,:],
                                )
    gxc_hvftrs_sub_g1[mod] = GC_matrix(
                                _genes_set1,
                                gxc_hvftrs_sub[mod].cell,
                                gxc_hvftrs_sub[mod].data.tocsr()[_genes_set1_index,:],
                                )
    
    print(mod, gxc_hvftrs_sub_g0[mod].data.shape, gxc_hvftrs_sub_g1[mod].data.shape, time.time()-ti)


# In[13]:


print(gxc_hvftrs_sub_g0.keys())
print(gxc_hvftrs_sub_g1.keys())


# In[14]:


resolutions = [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 20]
(
 res_nclsts, 
 res_mse_mean, res_mse_se, 
 res_mse_t_mean, res_mse_t_se, 
) = nfoldcv_scf(
            gxc_hvftrs_sub_g1, gxc_hvftrs_sub_g0, resolutions, k, 
            metas_sub, mods_selected, features_selected, settings,   
            ps, drop_npcs,
            cross_mod_distance_measure, knn, relaxation, n_cca,
            npc,
            output_pcX_all, output_cells_all, output_clst_and_umap,
            reduce_dim=0,
            nfolds=5, n_repeats=10)


# In[33]:


# Saving the objects:
with open(output_results, 'wb') as f: 
    pickle.dump((
                 res_nclsts, 
                 res_mse_mean, res_mse_se, 
                 res_mse_t_mean, res_mse_t_se, 
                ), f)


# In[34]:


# Getting back the objects:
with open(output_results, 'rb') as f: 
    (res_nclsts, 
     res_mse_mean, res_mse_se, 
     res_mse_t_mean, res_mse_t_se, 
    ) = pickle.load(f)


# In[35]:


output = output_figures.format('cluster_cv_sharey', 'pdf')
n = len(mods_selected)
nx = 3
ny = int((n+nx-1)/nx)
scale = 1
fig, axs = plt.subplots(ny, nx, figsize=(5*nx*scale,4*ny*scale), sharex=True, sharey=True)
axs = axs.flatten()
for i, (mod, ax) in enumerate(zip(mods_selected, axs)):
    base_level = np.min(res_mse_mean[mod])
    if i % nx == 0:
        ylabel = 'MSE +/- SEM\n(normalized)'
    else:
        ylabel = ''
    xlabel = ''
    plot_bi_cv_subfig(ax, res_nclsts, 
                      res_mse_mean[mod]/base_level, res_mse_se[mod]/base_level, 
                      res_mse_t_mean[mod]/base_level, res_mse_t_se[mod]/base_level, 
                      settings[mod].color, mod, 
                      xlabel=xlabel,
                      ylabel=ylabel
                     )
    ax.yaxis.set_major_locator(mtick.MaxNLocator(4))

fig.subplots_adjust(wspace=0.1, bottom=0.15)
fig.text(0.5, 0, 'Number of clusters', ha='center', fontsize=15)
fig.savefig(output, bbox_inches='tight')
plt.show()


# In[36]:


output = output_figures.format('cluster_cv_nosharey', 'pdf')
n = len(mods_selected)
nx = 3
ny = int((n+nx-1)/nx)
scale = 1
fig, axs = plt.subplots(ny, nx, figsize=(5*nx*scale,4*ny*scale), sharex=True, sharey=False)
axs = axs.flatten()
for i, (mod, ax) in enumerate(zip(mods_selected, axs)):
    base_level = np.min(res_mse_mean[mod])
    if i % nx == 0:
        ylabel = 'MSE +/- SEM\n(normalized)'
    else:
        ylabel = ''
    xlabel = ''
    plot_bi_cv_subfig(ax, res_nclsts, 
                      res_mse_mean[mod]/base_level, res_mse_se[mod]/base_level, 
                      res_mse_t_mean[mod]/base_level, res_mse_t_se[mod]/base_level, 
                      settings[mod].color, mod, 
                      xlabel=xlabel,
                      ylabel=ylabel
                     )
    ax.yaxis.set_major_locator(mtick.MaxNLocator(4))

fig.subplots_adjust(wspace=0.3, bottom=0.15)
fig.text(0.5, 0, 'Number of clusters', ha='center', fontsize=15)
fig.savefig(output, bbox_inches='tight')
plt.show()

