"""
"""

from __init__ import *

from scipy import sparse
from scipy import stats

def sparse_logcpm(gc_matrix, mode='logcpm', lib_size=[]):
    """
    """
    lib_size = np.array(lib_size)
    if np.size(lib_size) == 0:
        lib_size = gc_matrix.data.sum(axis=0)

    lib_size_inv = sparse.diags(np.ravel(1.0/(1e-7+lib_size)))
    cpm = (gc_matrix.data).dot(lib_size_inv*1e6).tocoo()

    if mode == 'logcpm':
        cpm.data = np.log10(cpm.data + 1)
    elif mode == 'cpm':
        pass

    gc_cpm = GC_matrix(
        gc_matrix.gene, 
        gc_matrix.cell, 
        cpm,
    )
    
    return gc_cpm

def get_index_from_array(arr, inqs, na_rep=-1):
    """Get index of array
    """
    arr = np.array(arr)
    arr = pd.Series(arr).reset_index().set_index(0)
    idxs = arr.reindex(inqs)['index'].fillna(na_rep).astype(int).values
    return idxs

def save_gc_matrix(gc_matrix, f_gene, f_cell, f_mat):
    """
    """
    sparse.save_npz(f_mat, gc_matrix.data)
    with open(f_gene, 'w') as f:
        f.write('\n'.join(gc_matrix.gene)+'\n')
    with open(f_cell, 'w') as f:
        f.write('\n'.join(gc_matrix.cell)+'\n')

def save_gc_matrix_methylation(gc_matrix, f_gene, f_cell, f_mat_mc, f_mat_c):
    """
    """
    sparse.save_npz(f_mat_mc, gc_matrix.data['mc'])
    sparse.save_npz(f_mat_c, gc_matrix.data['c'])
    with open(f_gene, 'w') as f:
        f.write('\n'.join(gc_matrix.gene)+'\n')
    with open(f_cell, 'w') as f:
        f.write('\n'.join(gc_matrix.cell)+'\n') 

def import_single_textcol(fname, header=None, col=0):
    return pd.read_csv(fname, header=header, sep='\t')[col].values

def export_single_textcol(fname, array):
    with open(fname, 'w') as f:
        f.write('\n'.join(array)+'\n')

def load_gc_matrix(f_gene, f_cell, f_mat):
    """
    """
    gene = import_single_textcol(f_gene)
    cell = import_single_textcol(f_cell)
    mat = sparse.load_npz(f_mat) 
    assert (len(gene), len(cell)) == mat.shape
    return GC_matrix(gene, cell, mat) 

def load_gc_matrix_methylation(f_gene, f_cell, f_mat_mc, f_mat_c):
    """
    """
    _gene = import_single_textcol(f_gene) 
    _cell = import_single_textcol(f_cell)
    _mat_mc = sparse.load_npz(f_mat_mc) 
    _mat_c = sparse.load_npz(f_mat_c) 
    gxc_raw = GC_matrix(_gene, _cell, 
                              {'c': _mat_c, 'mc': _mat_mc})
    return gxc_raw


