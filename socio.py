import numpy as np
from scipy.linalg import hankel
import matplotlib.pyplot as plt
from sympy.utilities.iterables import multiset_permutations


def socio_results(avg, stdev, number, plot=False):
    # set params
    p_average = avg
    p_std = stdev
    p_number = number
    
    # set more params out of func control
    param, meta = {}, {}
    error = 0.01
    param['min_number'] = 1
    param['max_number'] = 7
    
    # initialize
    meta['iteration'] = 0
    meta['history'] = list()
    
    # set starting vector
    total_sum = round(p_average * p_number)
    start_vec = list()
    for i in range(p_number):
        start_vec.append(np.round((total_sum - sum(start_vec))/ (p_number-i)))
    start_vec = sorted(start_vec)
    
    # compute results
    result, meta_new = socio(None, start_vec, param, meta, 0)
    result = result[np.max(result, axis=1) <= param['max_number'], :]
    result = result[np.min(result, axis=1) >= param['min_number'], :]
    
    # filter by STD
    if stdev is not None:
        result = result[np.abs(np.std(result, axis=1) - stdev) <= error, :]
    
    # create output
    output = {}
    output['result'] = result
    output['std'] = np.std(result, axis=1)
    output['avg'] = np.mean(result, axis=1)
    output['meta'] = meta_new
    
    if plot:
        if result.size > 0:
            plt.imshow(result, extent=[0, 1, 0, 1])
            plt.colorbar()
            plt.show()

    return output


def socio(mat, vec, param, meta_old, iteration):
    """
    Gets a matrix of vector permutations (initially empty) and a vector. 
    Returns all permutations of the vector such that:
        1. I's length is unchanged. 
        2. The sum of its values is unchanged. 
        3. Its values are bounded according to 'param' input argument.
    """
    # initialize - prepare new outputs
    if mat is None:
        mat_new = np.empty(shape=(0,len(vec)))
    else:
        mat_new = mat.copy()
    meta_new = meta_old.copy()
    meta_new['iteration'] += 1
    iteration += 1
    
    # get all permutations of vector. 
    mat_pos = all_perm(vec, param)
    
    # remove permutations that already exist in mat
    remove_idx = []
    for row_idx, row in enumerate(mat_pos):
        if is_vec_in_mat(row, mat_new):
            remove_idx.append(row_idx)
    mat_pos = np.delete(mat_pos, remove_idx, 0)
    
    # update vector matrix
    mat_new = np.vstack([mat_pos, mat_new])
    
    # update meta data - how many vectors were added. 
    meta_new['history'].append({'iteration': iteration, 'size': mat_pos.shape[0]}) # same
     # recursive loop - for each vector added, add all it's permutations as well. 
    for row in mat_pos:
        mat_new, meta_new = socio(mat_new, row, param, meta_new, iteration)
    
    return mat_new, meta_new


def all_pairs_unique_sum(vec):
    """gives all pairs (x, y) of items in vec."""
    
    # generate all pairs
    A = hankel(vec, np.roll(vec, 1))
    A = A[1:, :].flatten()
    n = int(len(A)/2)
    A = np.vstack([A[:n], 
                   np.tile(vec, int((len(A)/len(vec))))[:n]])
    # remove duplicates
    A.sort(axis=0)
    return np.unique(A.T, axis=0)


def is_vec_in_mat(vec, mat):
    """return 1 if any row in mat is vec, 0 otherwise. """
    if mat.shape[1] != len(vec):
        return 0
    if mat.shape[0] == 0:
        return 0
    return int(any(np.equal(mat,vec).all(1)))



def all_pairs_sum(x, y, param):
    """gives all pairs that are summed to x+y but are not (x, y)"""
    
    maxi, mini = param['max_number'], param['min_number']
    s = x+y
    num2exclude = min(x, y)
    # create all possible (with possible duplicates)
    vec = np.arange(mini, maxi+1)
    vec = np.vstack([vec, 
                     np.ones(shape=(1, len(vec))) * s - vec])
    # remove out of bounds (min, max) of param
    vec = vec[:, np.max(vec, axis=0) <= maxi]
    vec = vec[:, np.min(vec, axis=0) >= mini]
    # remove (x, y) pairs
    vec = vec[:, np.min(vec, axis=0) != num2exclude]
    
    if vec.size == 0:
        return vec
    
    # remove duplicates
    vec.sort(axis=0)
    return np.unique(vec, axis=1).T
    

def all_perm(vec, param):
    """gives all permutations of pairs in vec such that the sum is unchanged.
    will not give all vectors that have the same sum since it may need more than on 
        pair switch to obtain.
    """
    
    # get all pairs with unique sum
    pairs_u_sum = all_pairs_unique_sum(sorted(vec))
    
    # loop over each pair
    my_mat = [vec]
    for (x, y) in pairs_u_sum:
        # find pairs which have the same sum
        swap_mat = all_pairs_sum(x, y, param)
        if swap_mat.size == 0:
            continue
        # loop over alternative pairs, finding altenative vectors
        for (a, b) in swap_mat:
            new_vec = vec.copy()
            new_vec[np.argmax(new_vec==x)] = a
            new_vec[np.argmax(new_vec==y)] = b
            my_mat.append(sorted(new_vec))
            
    return np.array(my_mat)

