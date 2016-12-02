import networkx as nx
import numpy as np

strong_network = nx.Graph()
network = nx.Graph()

strong_network.add_edge(1,5)
strong_network.add_edge(1,6)
strong_network.add_edge(1,11)
strong_network.add_edge(5,7)
strong_network.add_edge(5,9)
strong_network.add_edge(6,9)
strong_network.add_edge(7,11)
strong_network.add_edge(9,11)

network.add_edge(1,2)
network.add_edge(1,3)
network.add_edge(1,5)
network.add_edge(1,6)
network.add_edge(1,7)
network.add_edge(1,8)
network.add_edge(1,9)
network.add_edge(1,11)
network.add_edge(2,9)
network.add_edge(3,7)
network.add_edge(3,9)
network.add_edge(5,7)
network.add_edge(5,9)
network.add_edge(6,7)
network.add_edge(6,9)
network.add_edge(7,8)
network.add_edge(7,9)
network.add_edge(7,11)
network.add_edge(8,9)
network.add_edge(9,11)

tpm_bulk=np.loadtxt('SRSF_bulk.tsv', delimiter=',', usecols=range(1,34))
tpm_single_cell=np.loadtxt('SRSF_scRNA.tsv', delimiter=',', usecols=range(1,2366))
tpm_bulk = tpm_bulk.T
tpm_single_cell = tpm_single_cell.T

def cleanMatrix(M):
	new_matrix = []
	for sample in M:
		if not 0 in sample:
			new_matrix.append(sample)
	return np.array(new_matrix)

def bucketByMedian(M, num_buckets=2):
	M = M.copy()
	buckets = []
	buckets.append(np.zeros(M.shape[1]))
	for i in range(1,num_buckets):
		buckets.append(np.percentile(M, 100.0*i/num_buckets, axis=0))
	for i in range(0,M.shape[0]):
		for j in range(0,M.shape[1]):
			max_bucket = 0
			for k in range(0,num_buckets):
				if M[i][j] >= buckets[k][j]:
					max_bucket = k
			M[i][j] = max_bucket
	return M

tpm_bulk = cleanMatrix(tpm_bulk)
median = bucketByMedian(tpm_bulk,num_buckets=5)