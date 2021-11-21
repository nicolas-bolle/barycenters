# Nicolas Bolle 11/19/21ish
# Implementing Wasserstein barycenter stuff on Google Quick Draw! data, using the Sinkhorn divergence

# Some references:
# https://arxiv.org/abs/1306.0895
# https://arxiv.org/abs/1310.4375
# https://arxiv.org/abs/1805.11897

# This file is for standalone methods and functions



## Imports
import numpy as np



### sinkhorn

## Inputs:
# r: length n numpy array giving a nonnegative measure on the set of n locations
# c: (m x k) numpy array, giving k nonnegative measures on the set of m locations
#    Must have sum(r) = [row sums of c]
# M: (n x m) matrix giving distances between the locations
# l: lambda parameter
# iterations: number of iterations to do

## Output:
# length k numpy array of the sinkhorn divergences

## Info
# Computes the Sinkhorn divergence between histograms r and c, using a distance matrix M and parameter l=lambda
# Useful for more user-friendly computations, when they haven't precomputed K for speed

def sinkhorn(r,c,M,l,iterations=20):
    K = np.exp(-l*M)
    return sinkhorn_mk(r,c,M,K,iterations)



### sinkhorn_mk

## Inputs:
# r: length n numpy array giving a nonnegative measure on the set of n locations
# c: (m x k) numpy array, giving k nonnegative measures on the set of m locations
#    Must have sum(r) = [row sums of c]
# M: (n x m) matrix giving distances between the locations
# K: (n x m) matrix, K = exp(-l * M) the elementwise exponential
# iterations: number of iterations to do

## Output:
# length k numpy array of the sinkhorn divergences

## Info
# Computes the Sinkhorn divergence between histograms r and c, using relevant matrices M and K
# Useful for more efficient computations, when K is precomputed

# FIXME: using a fixed number of iterations for now, add a stopping condition?
def sinkhorn_mk(r,c,M,K,iterations=20):
    
    ## Setup
    
    # Remove zeros in r to avoid division by zero
    I = r > 0
    r = r[I]
    M = M[I,:]
    K = K[I,:]
    
    # Initialize x, a (n x k) array
    n = len(r)
    k = np.shape(c)[1]
    # Note: the Sinkhorn paper makes the columns of x into probability distributions, but I ignore that
    x = np.ones((n,k))
    
    # Precompute a matrix product to speed up iteration
    # FIXME: does this actually give a speed up?
    R = np.diag(np.reciprocal(r)) @ K
    
    
    ## Iterate for x
    # FIXME: faster way of looping? i.e. not in Python. Would it give a speedup?
    for i in range(iterations):
        x = R @ (c * np.reciprocal(np.transpose(K) @ np.reciprocal(x)))
    
    
    ## Get u, v, which are (n x k) and (m x k) respectively
    u = np.reciprocal(x)
    v = c * np.reciprocal(np.transpose(K) @ u)
    
    
    ## Return the distance
    # Before the sum, we have a (n x k) array
    # So the sum is taken for each column
    return np.sum(u * ((K * M) @ v), axis = 0)












