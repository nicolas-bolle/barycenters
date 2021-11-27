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

def sinkhorn_mk(r,c,M,K,iterations=20):
    
    # Remove zeros in r to avoid division by zero
    I = r > 0
    r = r[I]
    M = M[I,:]
    K = K[I,:]
    
    # Reshape c
    if len(np.shape(c)) == 1:
        c = np.reshape(c,(len(c),1))
    
    # Run the iteration
    u,v = _uv_iteration(r,c,M,K,iterations)
    
    # Return the distance
    # Before the sum, we have a (n x k) array
    # So the sum is taken for each column
    return np.sum(u * ((K * M) @ v), axis = 0)



### _uv_iteration

## Inputs:
# r: length n numpy array giving a positive (!) measure on the set of n locations
# c: (m x k) numpy array, giving k nonnegative measures on the set of m locations
#    Must have sum(r) = [row sums of c]
# M: (n x m) matrix giving distances between the locations
# K: (n x m) matrix, K = exp(-l * M) the elementwise exponential
# iterations: number of iterations to do

## Output:
# tuple (u,v) of the vectors obtained from the iteration

## Info
# Helper function to do the Sinkhorn iteration

# FIXME: using a fixed number of iterations for now, add a stopping condition?
def _uv_iteration(r,c,M,K,iterations=20):
    
    ## Setup
    
    # Initialize x, an (n x k) array
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
    
    
    ## Return (u,v)
    return (u,v)



### Dsinkhorn_reg

## Inputs:
# r: length n numpy array giving a nonnegative measure on the set of n locations
# c: (m x k) numpy array, giving k nonnegative measures on the set of m locations
#    Must have sum(r) = [row sums of c]
# M: (n x m) matrix giving distances between the locations
# l: lambda parameter
# K: (n x m) matrix, K = exp(-l * M) the elementwise exponential
# iterations: number of iterations to do

## Output:
# (n x k) numpy array of the k gradients

## Info
# Computes the derivative of the regularized Sinkhorn divergence between histograms r and c, using relevant matrices M and K
# Useful for more efficient computations, when K is precomputed

def Dsinkhorn_reg(r,c,M,l,K,iterations=20):
    
    # Remove zeros in r to avoid division by zero
    I = r > 0
    r = r[I]
    M = M[I,:]
    K = K[I,:]
    
    # Reshape c
    if len(np.shape(c)) == 1:
        c = np.reshape(c,(len(c),1))
    
    # Run the iteration
    u,_ = _uv_iteration(r,c,M,K,iterations)
    
    # Turn this into alpha_*
    # Need the extra step of making sure sum(alpha) = 0
    alpha = np.zeros((len(I),np.shape(c)[1]))
    alpha_I = np.log(u) / l
    alpha_I = alpha_I - sum(alpha_I) / len(alpha_I)
    alpha[I,:] = alpha_I
    
    # Return
    return alpha



### Dsinkhorn

## Inputs:
# r: length n numpy array giving a nonnegative measure on the set of n locations
# c: (m x k) numpy array, giving k nonnegative measures on the set of m locations
#    Must have sum(r) = [row sums of c]
# M: (n x m) matrix giving distances between the locations
# l: lambda parameter
# K: (n x m) matrix, K = exp(-l * M) the elementwise exponential
# iterations: number of iterations to do

## Output:
# (n x k) numpy array of the k gradients

## Info
# Computes the derivative of the sharp Sinkhorn divergence between histograms r and c, using relevant matrices M and K
# Useful for more efficient computations, when K is precomputed

def Dsinkhorn(r,c,M,l,K,iterations=20):
    
    # Remove zeros in r to avoid division by zero
    I = r > 0
    r = r[I]
    M = M[I,:]
    K = K[I,:]
    
    # Reshape c
    if len(np.shape(c)) == 1:
        c = np.reshape(c,(len(c),1))
    
    # Run the iteration
    u,v = _uv_iteration(r,c,M,K,iterations)
    
    # Relevant sizes
    n = len(r)
    m,k = np.shape(c)
    
    ## For each column of u/v/c, do the computation
    # Could speed this up in the future, to fully vectorize things
    gradients = np.zeros((len(I),k))
    
    # Using p. 8 pseudocode in https://arxiv.org/abs/1805.11897
    # Slightly modified to make it correct, and replace all-ones-vectors-multiplication with sums along rows/columns
    for i in range(k):
        # T
        T = np.diag(u[:,i]) @ K @ np.diag(v[:,i])
        Tbar = T[:,:-1]
        
        # L
        L = T * M
        Lbar = L[:,:-1]
        
        # D1 and D2
        D1 = np.diag(np.sum(T,axis=1))
        D2 = np.diag(np.reciprocal(np.sum(Tbar,axis=0)))
        
        # H
        H = D1 - Tbar @ D2 @ np.transpose(Tbar)
        
        # f: minus sign because I think the pseudocode gives the descent direction (negative gradient)
        f = L @ np.ones(m) - Tbar @ D2 @ np.sum(Lbar,axis=0)
        
        # g: FIXME: scaling by lambda here? The formulas on p. 22/23 suggest that should happen
        g = l * np.linalg.solve(H,f)
        
        # Compute and save: modified to use np.mean() instead of np.sum(), since the sum of the output should be zero
        gradients[I,i] = g - np.mean(g) * np.ones(n)
    
    # Return all the gradients
    return gradients







