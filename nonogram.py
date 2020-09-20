from scipy.linalg import hankel
from sympy.utilities.iterables import multiset_permutations
import numpy as np
from fast_socio import find_vecs_that_sum  # faster alternative to socio
from socio import socio


def all_possible_vectors(vec_sum, vec_length, fast=True):
    """
    Return all possible sorted vectors such that:
        1. The sum of the vector is vec_sum
        2. The number of elements in the vector is vec_length
    """
    if fast:
        return np.array(find_vecs_that_sum(vec_length, vec_sum, 0, vec_sum))
    # set params
    p_number = vec_length
    
    # set more params out of fun control
    param, meta = {}, {}
    param['min_number'] = 0
    param['max_number'] = vec_sum
    
    # initialize
    meta['iteration'] = 0
    meta['history'] = list()
    
    # set starting vector
    total_sum = vec_sum
    start_vec = list()
    
    for i in range(p_number):
        start_vec.append(np.round((total_sum - sum(start_vec))/ (p_number-i)))
    start_vec.sort()
    
    # compute results
    result, meta_new = socio(None, start_vec, param, meta, 0)
    result = result[np.max(result, axis=1) <= param['max_number'], :]
    result = result[np.min(result, axis=1) >= param['min_number'], :]

    return result


def find_nonogram_options_matrix(row_len, block_sizes):
    # find all options for space values (sorted values)
    space_vec_length = len(block_sizes) + 1
    additional_spaces_needed = row_len + 1 - sum(block_sizes) - len(block_sizes)
#     space_values_list = all_possible_vectors(additional_spaces_needed, 
#                                              space_vec_length)
    space_values_list = find_vecs_that_sum(space_vec_length, additional_spaces_needed, 0, additional_spaces_needed)
    
#     # a trick that can be used to improve performance. But need to add ability to generate options based on board. 
#     # most likely using the existing spaces that have been determined - or a shorter vector to be solved. 
#     if len(space_values_list) > 100:
#         return None
    
    # find all configurations of spaces
    space_sizes = list()
    for space_vec in space_values_list:
        space_sizes += list(multiset_permutations(space_vec))
    space_sizes = np.array(space_sizes)
    space_sizes[:, 1:-1] += 1
    space_sizes

    # find all possible block starting positions
    loc_vecs = np.empty(shape=(space_sizes.shape[0], 
                               space_sizes.shape[1]+len(block_sizes)), 
                        dtype=int)
    loc_vecs[:, 0::2] = space_sizes
    loc_vecs[:, 1::2] = block_sizes
    loc_vecs = loc_vecs.cumsum(axis=1)[:, :-1:2]

    # create 0-1 matrix with all possible row vectors
    a_mat = np.zeros(shape=(loc_vecs.shape[0], row_len), dtype=int)
    for row_idx, loc_vec in enumerate(loc_vecs):
        for block_idx, start_pos in enumerate(loc_vec):
            a_mat[row_idx, 
                  start_pos:start_pos + block_sizes[block_idx]] = 1
    return a_mat


def vector_matrix_congruence(row_vec, mat):
    row_len = row_vec.size
    
    if mat is None:
        return row_vec, mat
    
    # transpose if input is a column vector
    if row_vec.shape == (row_len, 1):
        return propagate_nonogram_vector(row_vec.T, block_sizes).reshape(row_vec.shape)
    
    # row_vec must be a row vector
    assert ((row_vec.shape == (1, row_len)) 
            or (row_vec.shape == (row_len, ))), 'row_vec must be a row vector'


    # if row_vec is already known: return row_vec
    if not np.any(row_vec == -1):
        return row_vec, mat
    
    # remove row vectors that contradict the known 0-1 positions
    known_ones = np.clip(row_vec, 0, 1)
    known_zeros = np.clip(1-np.abs(row_vec), 0, 1)

    ones_match = (mat * known_ones).sum(axis=1) == known_ones.sum()
    mat = mat[ones_match, :]

    zeros_match = ((1-mat) * known_zeros).sum(axis=1) == known_zeros.sum()
    mat = mat[zeros_match, :]
    
    # find the positions where 0/1 must be
    new_vec = row_vec.reshape((1, row_len))
    new_vec[:, (mat==1).all(axis=0)] = 1
    new_vec[:, (mat==0).all(axis=0)] = 0
    
    return new_vec, mat


def propagate_nonogram_vector(row_vec, block_sizes):
    row_len = row_vec.size
    
    # transpose if input is a column vector
    if row_vec.shape == (row_len, 1):
        return propagate_nonogram_vector(row_vec.T, block_sizes).reshape(row_vec.shape)
    
    # row_vec must be a row vector
    assert ((row_vec.shape == (1, row_len)) 
            or (row_vec.shape == (row_len, ))), 'row_vec must be a row vector'

    # if row_vec is already known: return row_vec
    if not np.any(row_vec == -1):
        return row_vec
    
    # get matrix with all possible configurations of blocks
    a_mat = find_nonogram_options_matrix(row_len, block_sizes)
    
    # return the congruence of the matrix 
    #     with the input "known" vector as the new known.
    new_vec, _ = vector_matrix_congruence(row_vec, a_mat)
    return new_vec



def propagate_board(board_mat, options_dict_r, options_dict_c, verbose=False, iterations=100):
    
    for k in range(iterations):
        squares_left = np.sum(np.sum((board_mat == -1))) / board_mat.size
        if verbose: 
            print(f"Iteration {k}\t - {100*squares_left:0.4}% of board left")
        
        row_num, col_num = board_mat.shape

        if squares_left == 0:
            return board_mat, True, options_dict_r, options_dict_c

        for i in range(row_num):
            board_mat[i, :], options_dict_r[i] = vector_matrix_congruence(
                board_mat[i, :], options_dict_r[i])        
        for i in range(col_num):
            board_mat[:, i], options_dict_c[i] = vector_matrix_congruence(
                board_mat[:, i], options_dict_c[i])
                
        new_squares_left = np.sum(np.sum((board_mat == -1))) / board_mat.size

        if new_squares_left == squares_left:
            return board_mat, False, options_dict_r, options_dict_c
    print('not enough iterations')
    return board_mat, False, options_dict_r, options_dict_c