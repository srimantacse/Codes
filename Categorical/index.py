import math
import sys
from numpy import array
from utils import list_max, list_min
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_samples, silhouette_score
from utils import list_max, list_min, multiIndexOf, unique
from scipy.spatial.distance import cdist
from math import*
from decimal import Decimal
import copy
import numpy as np
import scipy.spatial

"""
	Various Cluster validity indices
	Jm: FCM Cost Function
	Aplha:
	Beta:
	ARI:
	DB:
	DUNN:
	SILHOUETTE:
	MINKOWSKI:
	XB:
	JACCARD: 	
	PC: Partition Coefficient
	NPC: Normalized Partition Coefficient
	FHV: Fuzzy Hypervolume
	FS: Fukuyama-Sugeno Index	
	BH: Beringer-Hullermeier Index
	BWS: Bouguessa-Wang-Sun index
"""

"""
	All the Helper Functions
"""
def Jm(data, center, U):
	dist = cdist(centers, data, metric='sqeuclidean')
	return np.sum(U * dist)

def relabeling(actualClass, predictedClass):
	# actualClass      = Reference cluster label vector.
	# predictedClass   = Query cluster label vector.
	# mappedClass      = Query vector after mapping.
	
	mappedClass = predictedClass
	
	minLabel    = list_min(predictedClass)
	maxLabel    = list_max(predictedClass)

	for i in range(minLabel, maxLabel):
		a = multiIndexOf(predictedClass, i);
		b = actualClass(a);
		x = unique(b);
		v = i;
		maxm = -9;
		for j in range(0, len(x) - 1):
			t = x[j];
			y = len(multiIndexOf(b,t));
			
			if y > maxm:
				v = t
				maxm = y;
    
		mappedClass[a] = v;

	return mappedClass

def clusterMovements(cluster1, cluster2):
	# cluster1: cluster centers set 1
	# cluster2: cluster centers set 2
	distance = euclidean_distances(cluster1, cluster2)
	result = sum(sum(distance)) / 2
	
	ret = '{:0,.2f}'.format(float(result))
	return ret

def nth_root(value, n_root):
    root_value = 1/float(n_root)
    return round (Decimal(value) ** Decimal(root_value),3)

def pairwise_squared_distances(A, B):
    return scipy.spatial.distance.cdist(A, B)**2

def calculate_covariances(x, u, v, m):
    c, n = u.shape
    d = v.shape[1]
    
    um = u**m

    covariances = np.zeros((c, d, d))

    for i in range(c):
        xv = x - v[i]
        uxv = um[i, :, np.newaxis]*xv
        covariances[i] = np.einsum('ni,nj->ij', uxv, xv)/np.sum(um[i])
    
    return covariances

def calculate_dissimilarity(Z, X):
    """
    Calculates disssimilarity between any 2 records supplied by using Simple Matching Dissimilarity Measure algorithm.
    :param Z: Record 1
    :param X: Record 2
    :return: Dissimilarity between Z and X
    """
    m = len(Z)

    dissimlarity = 0

    for j in range(m):
        if Z[j] != X[j]:
            dissimlarity += 1

    return dissimlarity
	
"""
	All the Index Function
"""
def alpha(cluster, data):
	# data: Data points
	# cluster: cluster centers
	distance = euclidean_distances(data, cluster)
	
	resultedClassLabel = []
	for i in range(len(data)):
		idx, val = list_min(distance[i])
		resultedClassLabel.append(idx)
		
	result = 0.0
	for i in range(len(data)):
		dist = euclidean_distances(array(data.loc[i]).reshape(1,-1), array(cluster[resultedClassLabel[i]]).reshape(1,-1))
		result = result + (dist**2)
		
	ret = '{:0,.2f}'.format(float(result / len(data)))
	#ret = '{:0,.2f}'.format(result / (len(unique(resultedClassLabel))))
	return ret
	
def beta(cluster, data):
	# data: Data points
	# cluster: cluster centers
	distance = euclidean_distances(data, cluster)
	
	resultedClassLabel = []
	for i in range(len(data)):
		idx, val = list_min(distance[i])
		resultedClassLabel.append(idx)
	
	result = 0.0
	for i in unique(resultedClassLabel):
		indexList = multiIndexOf(resultedClassLabel, i)
		elementCount = len(indexList)
		if elementCount <= 1:
			continue
			
		#_data = data[indexList] 
		#_data = list(map(list(data).__getitem__, indexList))
		_data = [data.loc[i] for i in indexList]
		_distance = euclidean_distances(_data, _data) ** 2
		result = result + (sum(sum(_distance)) / (elementCount * (elementCount - 1)))
	
	#ret = "{0:.3f}".format(float(result / (len(unique(resultedClassLabel)))))
	ret = '{:0,.2f}'.format(result / (len(unique(resultedClassLabel))))
	return ret

def ari(actualLabel, predictedLabel):
	# actualLabel: Ground truth
	# predictedLabel : Predicted labels for each sample.
	return adjusted_rand_score(actualLabel, predictedLabel)

def db(X, Y, Z):
    k = Z.__len__()

    dist_i = []

    for ii in range(k):
        centroid = Z[ii]
        points = [X[i] for i in range(len(Y)) if Y[i] - 1 == ii]
        distance = 0

        for jj in points:
            distance += calculate_dissimilarity(centroid, jj)

        if len(points) == 0:
            dist_i.append(0.0)
        else:
            dist_i.append(round(distance * 1.0 / len(points), 4))

    D_ij = []

    for ii in range(k):
        D_i = []
        for jj in range(k):
            if ii == jj or calculate_dissimilarity(Z[ii], Z[jj]) == 0:
                D_i.append(0.0)
            else:
                D_i .append((dist_i[ii] + dist_i[jj]) * 1.0 / calculate_dissimilarity(Z[ii], Z[jj]))
        D_ij.append(D_i)

    db_index = 0

    for ii in range(k):
        db_index += max(D_ij[ii])

    db_index *= 1.0
    db_index /= k

    return db_index

def dunn(X, Y, Z):
    k = Z.__len__()

    mean_distance_i = []

    for ii in range(k):
        centroid = copy.copy(Z[ii])
        points = [X[i] for i in range(len(Y)) if Y[i] - 1 == ii]

        if len(points) == 0:
            mean_distance_i.append(0.0)
        else:
            distance = 0
            for jj in points:
                distance += calculate_dissimilarity(centroid, jj)
            mean_distance_i.append(round(distance * 1.0 / len(points), 4))

    distance_ij = []

    for ii in range(k):
        for jj in range(ii+1, k):
            distance_ij.append(calculate_dissimilarity(Z[ii], Z[jj]))

    dunn_index = min(distance_ij) * 1.0 / max(mean_distance_i)

    return dunn_index
	
def dunn_1(labels, distances):
	# points: Data points
	# predictedLabel : Predicted labels for each sample.
	
	"""
    Dunn index for cluster validation (the bigger, the better)
    
    .. math:: D = \\min_{i = 1 \\ldots n_c; j = i + 1\ldots n_c} \\left\\lbrace \\frac{d \\left( c_i,c_j \\right)}{\\max_{k = 1 \\ldots n_c} \\left(diam \\left(c_k \\right) \\right)} \\right\\rbrace
    
    where :math:`d(c_i,c_j)` represents the distance between
    clusters :math:`c_i` and :math:`c_j`, given by the distances between its
    two closest data points, and :math:`diam(c_k)` is the diameter of cluster
    :math:`c_k`, given by the distance between its two farthest data points.
    
    The bigger the value of the resulting Dunn index, the better the clustering
    result is considered, since higher values indicate that clusters are
    compact (small :math:`diam(c_k)`) and far apart.
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    
    .. [Kovacs2005] Kovács, F., Legány, C., & Babos, A. (2005). Cluster validity measurement techniques. 6th International Symposium of Hungarian Researchers on Computational Intelligence.
    """

	labels = normalize_to_smallest_integers(labels)

	unique_cluster_distances = np.unique(min_cluster_distances(labels, distances))
	max_diameter = max(diameter(labels, distances))

	if np.size(unique_cluster_distances) > 1:
		return unique_cluster_distances[1] / max_diameter
	else:
		return unique_cluster_distances[0] / max_diameter
	
def silhouette(pairWisePointDistance, clusterLabels):
	# pairWisePointDistance: Array of pairwise distances between samples, or a feature array.
	# clusterLabels: 		 Predicted labels for each sample.
	return silhouette_score(pairWisePointDistance, clusterLabels)
	
def minkowski(x,y,p_value):
 
    return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)
	
def jaccard_similarity(x,y):
 
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

def xb(x, u, v, m):
    n = x.shape[0]
    c = v.shape[0]

    um = u**m
    
    d2 = pairwise_squared_distances(x, v)
    v2 = pairwise_squared_distances(v, v)
    
    v2[v2 == 0.0] = np.inf

    return np.sum(um.T*d2)/(n*np.min(v2))
	
def pc(x, u, v, m):
    c, n = u.shape
    return np.square(u).sum()/n

def npc(x, u, v, m):
    n, c = u.shape
    return 1 - c/(c - 1)*(1 - pc(x, u, v, m))

def fhv(x, u, v, m):
    covariances = calculate_covariances(x, u, v, m)
    return sum(np.sqrt(np.linalg.det(cov)) for cov in covariances)

def fs(x, u, v, m):
    n = x.shape[0]
    c = v.shape[0]

    um = u**m

    v_mean = v.mean(axis=0)

    d2 = pairwise_squared_distances(x, v)
    
    distance_v_mean_squared = np.linalg.norm(v - v_mean, axis=1, keepdims=True)**2

    return np.sum(um.T*d2) - np.sum(um*distance_v_mean_squared)

def bh(x, u, v, m):
    n, d = x.shape
    c = v.shape[0]

    d2 = pairwise_squared_distances(x, v)
    v2 = pairwise_squared_distances(v, v)
    
    v2[v2 == 0.0] = np.inf

    V = np.sum(u*d2.T, axis=1)/np.sum(u, axis=1)

    return np.sum(u**m*d2.T)/n*0.5*np.sum(np.outer(V, V)/v2)

def bws(x, u, v, m):
    n, d = x.shape
    c = v.shape[0]

    x_mean = x.mean(axis=0)

    covariances = calculate_covariances(x, u, v, m)

    sep = np.einsum("ik,ij->", u**m, np.square(v - x_mean))
    comp = sum(np.trace(covariance) for covariance in covariances)

    return sep/comp