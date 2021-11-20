# Nicolas Bolle 11/19/21ish
# Implementing Wasserstein barycenter stuff on Google Quick Draw! data, using the Sinkhorn divergence

# Some references:
# differential properties of sinkhorn
# fast computation of wasserstein barycenters
# sinkhorn distances: lightspeed computation

# This file is for standalone methods and functions


# Imports
import numpy as np


# Compute the Sinkhorn divergence between histograms p1 and p2, using a distance matrix M and parameter l=lambda
# For more user-friendly computations, when they only have M and lambda
# FIXME: reduce sizes of M and K based on supports of p1, p2?
def sinkhorn(p1,p2,M,l):
    K = np.exp(-l*M)
    return sinkhorn_mk(p1,p2,M,K)


# Compute the Sinkhorn divergence between histograms p1 and p2, using relevant matrices M and K
# For more efficient computations, when K is precomputed
def sinkhorn_mk(p1,p2,M,K):
    # FIXME: using a fixed number of iterations for now, add a stopping condition
    x