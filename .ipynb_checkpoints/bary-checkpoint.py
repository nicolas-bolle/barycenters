# Nicolas Bolle 11/19/21ish
# Implementing Wasserstein barycenter stuff on Google Quick Draw! data, using the Sinkhorn divergence

# Some references:
# https://arxiv.org/abs/1306.0895
# https://arxiv.org/abs/1310.4375
# https://arxiv.org/abs/1805.11897

# This file is for standalone methods and functions



## Imports

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bars

# Data: https://www.tensorflow.org/datasets
import tensorflow_datasets as tfds



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



### load_MNIST

## Inputs:
# N: number of digits to grab

## Output:
# M: distance matrix for the 784-element histogram
# X: (784 x N) numpy array, where columns are (normalized) histograms for each digit
# y: length 784 numpy vector of the labels for the columns of X

## Info
# Loads in the MNIST data for running Sinkhorn stuff on it

# FIXME: randomize which images we get
def load_MNIST(N=1000):
    
    ## Get M

    n = 28 * 28

    # To make sure I don't mess up indexing things, I'll set up a list of locations and reshape it into a matrix
    # So when I calculate a pairwise distance in the matrix, I can easily associate it to the location in the vector
    locations_vec = np.array(range(n))
    locations_arr = np.reshape(locations_vec, (28,28))

    M = np.zeros((n,n))
    # Having 4 "for" loops is a bit embarassing, but I'm not trying to think too hard right now
    for i1 in range(28):
        for j1 in range(28):
            for i2 in range(28):
                for j2 in range(28):
                    M[locations_arr[i1,j1], locations_arr[i2,j2]] = np.sqrt((i1-i2)**2 + (j1-j2)**2)
    
    
    ## Load MNIST data
    ds = tfds.load('mnist', split='train')
    
    
    ## Convert to a numpy array of histograms
    
    if N > 60000:
        N = 60000
        print("Can load at most 60,000 digits! Requested number of digits capped at 60,000.")

    # Allocate memory
    X = np.zeros((n,N))
    y = np.zeros(N)

    # Iterate to populate X
    i = 0
    for ex in ds.take(N):
        # Make sure to normalize too!
        aux = np.ravel(ex['image'].numpy())
        X[:,i] = aux / sum(aux)
        y[i]   = ex['label'].numpy()
        i = i + 1
    
    
    ## Return
    return (M,X,y)



### plot_digits

## Inputs:
# X: (784 x N) numpy array of histograms of digits to plot
# width: number of columns to have in the array of subplots

## Output:
# None. Just prints a plot of the digits.

## Info
# Plots histograms of digits

def plot_digits(X, width=5):
    shape = np.shape(X)
    
    if len(shape) == 1 or shape[1] == 1:
        # Single digit plot
        img = np.reshape(X, (28,28))
        plt.imshow(img)
        plt.colorbar()
        
    else:
        # Multi digit plot, I'll do rows of 5
        N = shape[1]
        
        rows = int(np.ceil(N/width))
        
        fig, axs = plt.subplots(rows,width,figsize=(15,5))

        # Iterate over the plotting locations
        i = 0
        for ax in axs.ravel():
            if i >= N:
                # Empty plot
                ax.axis('off')
            else:
                # Plot something
                ax.imshow(np.reshape(X[:,i], (28,28)))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(str(i+1))
                i = i + 1

                

### sinkhorn_barycenter

## Inputs:
# M: (n x n) numpy array of distances for the histograms
# X: (n x N) numpy array of histograms of digits to average

# Optional ones:
# r: the initial histogram
# noise: how much of the total mass should be dedicated to "noise"; so a number [0,1)
# steps: how many gradient descent steps to do
# eta: the amount the (1-normalized) gradient is scaled by
# l: lambda for the individual DSinkhorn computations
# iterations: number of iterations to do in the DSinkhorn computations


## Output:
# r: histogram giving the barycenter
# R: (n x (steps+1)) numpy array giving the histograms after each step
# G: (n x (steps+1)) numpy array giving the (scaled) gradients used at each step

## Info
# Computes the Sinkhorn barycenter of some digits, using the gradient descent method with sharp Sinkhorn divergence

def sinkhorn_barycenter(M, X, r = None, noise = 0.01, steps = 20, eta = 0.5, l = 10, iterations = 20):
    
    # Get relevant sizes
    n, N = np.shape(X)
    
    # Initialize r (with unit mass)
    if r is None:
        r = np.ones(n) / n
    else:
        r = r / np.sum(r)
    
    # Rescale X to have unit mass and some "noise"
    # So calculate how much mass is in each, add a proportionate amount of noise, and normalize
    # The amount of noise to add is calculated so that after normalizing, the amount of "noise" matches the user input amount
    masses = np.reshape(np.sum(X, axis = 0), (1,N))
    X = X + ((1 / (1-noise) - 1) / n) * masses
    masses = np.reshape(np.sum(X, axis = 0), (1,N))
    X = X / masses
    
    # I scale the gradient to have 1-norm of one, which seems reasonable enough
    # Would still like to know why the gradients I get are so big

    # Kernel
    K = np.exp(-l*M)

    # For keeping track of histograms and gradients
    R = np.zeros((n,steps+1))
    R[:,0] = r
    G = np.zeros((n,steps+1)) # The zero index will be unused

    # Iterate
    for i in tqdm(range(steps), desc="Gradient descent progress"):
        # FIXME add noise to r here?
        
        # Gradient
        gradients = Dsinkhorn(r,X,M,l,K,iterations)

        # Average together the gradients, reformat to match shape of r, rescale, and move opposite that direction
        g = np.reshape(np.mean(gradients, axis=1),np.shape(r))
        g = (eta / np.sum(np.abs(g))) * g
        r = r - g

        # Keep it nonnegative and unit mass
        r = r * (r>0)
        r = r / sum(r)

        # Save it
        R[:,i+1] = r
        G[:,i+1] = g
    
    
    ## Return
    return (r,R,G)



### _sinkhorn_barycenter

## Inputs:
# M: (n x n) numpy array of distances for the histograms
# l: lambda
# K: (n x n) numpy array giving np.exp(-l*M)
# X: (n x N) numpy array of histograms of digits to average, histograms normalized

# Optional ones:
# r: the initial histogram (normalized)
# eta: the amount the (1-normalized) gradient is scaled by
# iter_grad: how many gradient descent steps to do
# iter_Dsink: number of iterations to do in the DSinkhorn computations


## Output:
# r: histogram giving the barycenter

## Info
# Utility version of sinkhorn_barycenter():
# - No progress bar
# - Single output
# - Takes precomputed K
# - Does not add noise to X

def _sinkhorn_barycenter(M, l, K, X, r = None, eta = 0.5, iter_grad = 20, iter_Dsink = 20):
    
    # Get relevant sizes
    n, N = np.shape(X)
    
    # Initialize r (with unit mass)
    if r is None:
        r = np.ones(n) / n
    
    # I scale the gradient to have 1-norm of one, which seems reasonable enough
    # Would still like to know why the gradients I get are so big

    # Iterate
    for i in range(iter_grad):
        # FIXME add noise to r here?
        
        # Gradient
        gradients = Dsinkhorn(r,X,M,l,K,iter_Dsink)

        # Average together the gradients, reformat to match shape of r, rescale, and move opposite that direction
        g = np.reshape(np.mean(gradients, axis=1),np.shape(r))
        g = (eta / np.sum(np.abs(g))) * g
        r = r - g

        # Keep it nonnegative and unit mass
        r = r * (r>0)
        r = r / sum(r)
    
    
    ## Return
    return r



### k_means_sinkhorn_barycenter

## Inputs:
# M: (n x n) numpy array of distances for the histograms
# X: (n x N) numpy array of histograms of digits to cluster
# k: number of clusters to make

# Optional ones:
# p: the probability of "failure" when subsetting data for quick initilization
# noise: how much of the total mass should be dedicated to "noise"; so a number [0,1)
# eta: the amount the (1-normalized) gradient is scaled by
# l: lambda for the individual DSinkhorn computations
# p: the probability that not all digits will be represented when doing k-means++ initialization. Smaller = longer computation time.
# iter_sink: number of iterations to do in the Sinkhorn computations
# iter_Dsink: number of iterations to do in the DSinkhorn computations
# iter_grad: number of gradient descent steps to do
# iter_lloyd: number of iterations to do of Lloyd's algorithm
# I: manually initialize the representatives for initial clusters of seeds, by passing it columns of X


## Output:
# c: length N vector of cluster labels for the histograms
# r: (n x k) histograms giving the centers of clusters
# R: (n x k x (iter_lloyd+1)) numpy array giving the cluster center histograms after each step

## Info
# Clusters digits using Lloyd's algorithm (k-means++ initialization) with Sinkhorn divergence as the metric, and averages digits using Sinkhorn barycenters computed with sharp Sinkhorn divergence gradient descent. In Lloyd, I take distance to the first power as the quantity of interest (not 2nd power like usual Euclidean distance versions use).

def k_means_sinkhorn_barycenter(M, X, k, p = 0.01, noise = 0.01, eta = 1, l = 10, iter_sink = 20, iter_Dsink = 20, iter_grad = 4, iter_lloyd = 10, I = None):
    
    ## Get relevant sizes
    n, N = np.shape(X)
    
    
    ## Rescale X to have unit mass and some "noise"
    # So calculate how much mass is in each, add a proportionate amount of noise, and normalize
    # The amount of noise to add is calculated so that after normalizing, the amount of "noise" matches the user input amount
    masses = np.reshape(np.sum(X, axis = 0), (1,N))
    X = X + ((1 / (1-noise) - 1) / n) * masses
    masses = np.reshape(np.sum(X, axis = 0), (1,N))
    X = X / masses
    
    
    ## Kernel
    K = np.exp(-l*M)
    
    
    ## Initialize r as k random digits
    #I = np.random.choice(N, size = k, replace = False)
    #r = X[:,I]
    
    
    ## Initialize with k-means++, but picking from a subsample of the full data to reduce computation time
    # Size of subsample is chosen so that probability any of the k clusters isn't represented is less than p
    # The formula is given by setting up a union bound:
    #  P(fail) <= k * [Single digit missing] = k * (1-1/k)^samples ~ k e^(-samples/k) <= p
    
    print('Initializing...')
    
    if I is None:
    
        # Number of samples to use
        num_samples = min(int(np.ceil(k * np.log(k/p))), N)

        # Subset data for efficient k-means++ initialization
        Xp = X[:, np.random.choice(N, size = num_samples, replace = False)]

        # Pick the first digit ("I" is a list)
        I = [np.random.choice(num_samples)]

        # Iterate the k-means++ process to pick digits 2, ..., k
        for i in range(k-1):
            # Distances between picked digits and possible options
            D = _pairwise_distances(Xp[:,I], Xp, M, K, iter_sink)

            # Pick the smallest distance for each digit
            d = np.min(D, axis=0)

            # Pick the new digit use these distances as weights
            new = np.random.choice(num_samples, p = d / sum(d))

            # Add it to our list of digits
            I.append(new)

        # Define r to have columns giving the histograms of these digits
        r = Xp[:,I]
        
    else:
        
        if len(I) != k:
            print("Error: the list I must have length k")
            return
        else:
            r = X[:,I]
    
    print('Done initializing')
    
    
    ## For keeping track of histograms
    R = np.zeros((n,k,iter_lloyd+1))
    R[:,:,0] = r
    
    
    ## Iterate
    for i in tqdm(range(iter_lloyd), desc="Lloyd's algorithm progress"):
        
        ## Compute clusters of digits

        # Distances, using a helper function
        D = _pairwise_distances(r, X, M, K, iter_sink)
        
        # Pick clusters
        c = np.argmin(D, axis=0)
        
        
        ## Average together clusters to get barycenters
        
        for j in range(k):
            # Pick out the digits in this cluster
            I = (c == j)
            
            # Compute the barycenter for this cluster
            r[:,j] = _sinkhorn_barycenter(M, l, K, X[:,I], eta = eta, iter_Dsink = iter_Dsink, iter_grad = iter_grad)

            
        ## Save it
        R[:,:,i+1] = r
    
    
    ## Return
    return (c,r,R)



### _pairwise_distances

## Inputs:
# A: (n x a) numpy array of histograms
# B: (n x b) numpy array of histograms -> assuming b > a for efficiency
# M: (n x n) numpy array of distances for the histograms
# K: np.exp(-l*M)
# iter_sink: number of iterations for computing Sinkhorn divergences

## Output:
# D: (a x b) numpy array of distances between histograms in A and B

## Info
# Helper function to compute distances between cluster representatives and data points

def _pairwise_distances(A, B, M, K, iter_sink):
    
    # Sizes
    n, a = np.shape(A)
    n, b = np.shape(B)
    
    # Initialize D
    D = np.zeros((a,b))
    
    for i in range(a):
        D[i,:] = sinkhorn_mk(A[:,i] ,B ,M , K, iterations=iter_sink)
    
    return D


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