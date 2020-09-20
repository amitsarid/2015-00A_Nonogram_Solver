
def find_vecs_that_sum(vec_len, total, floor, ceil):
    result = find_vecs_that_sum_recursive(vec_len, total, floor, ceil)
    if result is None:
        return [[]]
    return result

def find_vecs_that_sum_recursive(vec_len, total, floor, ceil):
    
    # is an empty vector required?
    if vec_len == 0:
        # if the total is reached return an empty list, 
        #     otherwise there's no solution (None)
        if total == 0: 
            return [[]]
        else:
            return None
    
    assert vec_len > 0, 'vector length must be non-negative'
    
    # no solution if ceil < floor
    if ceil < floor:
        return None
        
    # no solution if the total can't be reached with the highest values
    if total > ceil * vec_len:
        return None
    
    # no solution if the total can't be reached with the smallest values
    if total < floor * vec_len:
        return None
    
    # trivial solution for vector length of size 1
    if vec_len == 1:
        if floor <= total <= ceil:
            return [[total]]
        else: 
            return None
    
    # otherwise, solve by recursion. 
    #     try the ceil 0 times, 1 time, 2 times, etc. 
    #     and aim to complete the rest of the vector.
    result = list()
    for i in range(vec_len+1):
        vec_head = [ceil] * i

        rest_of_vec = find_vecs_that_sum_recursive(
                                         vec_len-len(vec_head), 
                                         total-sum(vec_head), 
                                         floor, 
                                         ceil-1)
        if rest_of_vec is None:
            continue

        result += sorted([vec_head + v for v in rest_of_vec])
    
    return result


def pairs_that_sum(total, floor, ceil):
    # create all possible (with possible duplicates)
    vec = np.arange(floor, ceil+1)
    vec = np.vstack([vec, 
                     np.ones(shape=(1, len(vec))) * total - vec])
    # remove out of bounds (min, max) of param
    vec = vec[:, np.max(vec, axis=0) <= ceil]
    vec = vec[:, np.min(vec, axis=0) >= floor]
    
    # remove duplicates
    vec.sort(axis=0)
    return np.unique(vec, axis=1).T