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
        T = np.reshape(u[:,i],(n,1)) * K * np.reshape(v[:,i],(1,m))
        #T = np.diag(u[:,i]) @ K @ np.diag(v[:,i])
        Tbar = T[:,:-1]
        
        # L
        L = T * M
        Lbar = L[:,:-1]
        
        # D1 and d2
        D1 = np.diag(np.sum(T,axis=1))
        d2 = np.reciprocal(np.sum(Tbar,axis=0))
        
        # TbarD2, since we'll use it twice
        TbarD2 = Tbar * np.reshape(d2, (1, m-1))
        
        # H
        H = D1 - TbarD2 @ np.transpose(Tbar)
        
        # f: minus sign because I think the pseudocode gives the descent direction (negative gradient)
        f = np.sum(L, axis=1) - TbarD2 @ np.sum(Lbar,axis=0)
        
        # g: scaling by lambda here since the formulas on p. 22/23 suggest it
        g = l * np.linalg.solve(H,f)
        
        # Compute and save: uses np.mean() since the sum of the output should be zero
        gradients[I,i] = g - np.mean(g) * np.ones(n)
    
    # Return all the gradients
    return gradients










# Attempt at vectorizing: it's much slower :(
def Dsinkhorn_prime(r,c,M,l,K,iterations=20):
    
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
    
    # Using p. 8 pseudocode in https://arxiv.org/abs/1805.11897, tensorized
    # Slightly modified to make it correct
    
    # T: n x m x k
    # Formula: T_ijk = u_ik K_ij v_jk
    # Implemented easily by making everything a tensor and using broadcasting
    T = np.reshape(u,(n,1,k)) * np.reshape(K,(n,m,1)) * np.reshape(v,(1,m,k))
    Tbar = T[:,:-1,:]
    
    # L: n x m x k
    L = T * np.reshape(M,(n,m,1))
    Lbar = L[:,:-1,:]
    
    # D: n x k and (m-1) x k since I'm just looking at the diagonal
    d1 = np.sum(T,axis=1)
    d2 = np.reciprocal(np.sum(Tbar,axis=0))
    
    # TbarD2: n x (m-1) x k intermediate that I'll use soon
    # Avoids making D2 as a diagonal matrix, and instead does the multiplication as elementwise multiplication + broadcasting
    TbarD2 = Tbar * np.reshape(d2,(1,(m-1),k))
    
    # Give up on vectorizing H, f, g, lol
    g = np.zeros((n,k))
    for i in range(k):
        print(i)
        H = np.diag(d1[:,i]) - TbarD2[:,:,i] @ np.transpose(Tbar[:,:,i])
        f = np.sum(L[:,:,i], axis = 1) - TbarD2[:,:,i] @ np.sum(Lbar[:,:,i], axis = 0)
        g[:,i] = l * np.linalg.solve(H,f)
    
    ## Old code for H, f, g
    
    # This is by far the slowest step
    #aux = np.einsum('ilk,jlk->ijk', TbarD2, Tbar)
    #print('stop')
    
    # H: n x n x k
    # 1st term: turning d1 into a diagonal matrix on each of the k slices
    # 2nd term: formula aux_ijk = sum_l TbarD2_ilk Tbar_jlk
    #H = np.apply_along_axis(np.diag, 0 , d1) - aux
    
    # f: n x k
    #f = np.sum(L,axis=1) - np.einsum('ilj,lj->ij', TbarD2, np.sum(Lbar, axis=0))
    
    # g: n x k
    # FIXME: try to vectorize this? Is it even worth it?
    #g = np.zeros((n,k))
    #for i in range(k):
    #    g[:,i] = l * np.linalg.solve(H[:,:,i],f[:,i])
    
    # gradients: [original length of r] x k
    # Sum of each column is zero
    gradients = np.zeros((len(I),k))
    gradients[I,:] = g - np.outer(np.ones(n), np.mean(g, axis = 0))
    
    # Return all the gradients
    return gradients