import networkx as nx
import numpy as np

strong_network = nx.Graph()
network = nx.Graph()

strong_network.add_edge(0,1)
strong_network.add_edge(0,2)
strong_network.add_edge(0,5)
strong_network.add_edge(1,3)
strong_network.add_edge(1,4)
strong_network.add_edge(2,4)
strong_network.add_edge(3,5)
strong_network.add_edge(4,5)

network.add_edge(0,1)
network.add_edge(0,2)
network.add_edge(0,3)
network.add_edge(0,4)
network.add_edge(0,5)
network.add_edge(0,6)
network.add_edge(0,7)
network.add_edge(0,8)
network.add_edge(1,7)
network.add_edge(2,5)
network.add_edge(2,7)
network.add_edge(3,5)
network.add_edge(3,7)
network.add_edge(4,5)
network.add_edge(4,7)
network.add_edge(5,6)
network.add_edge(5,7)
network.add_edge(5,8)
network.add_edge(6,7)
network.add_edge(7,8)

tpm_bulk=np.loadtxt('SRSF_bulk.tsv', delimiter=',', usecols=range(1,34))
tpm_single_cell=np.loadtxt('SRSF_scRNA.tsv', delimiter=',', usecols=range(1,2366))
tpm_bulk = tpm_bulk.T
tpm_single_cell = tpm_single_cell.T


def cleanMatrix(M):
	"""
	Removes all samples with dropout (ie 0 in data matrix)
	"""
	new_matrix = []
	for sample in M:
		if not 0 in sample:
			new_matrix.append(sample)
	return np.array(new_matrix)

def bucketByMedian(M, num_buckets=2):
	"""
	Buckets each sample by its quartile value. 
	Ie if we had 1 dimensional data matrix [0,1,2,3,4,10] -> [0,0,1,1,2,2] for 3 buckets
	"""
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

def bucketByRange(M, num_buckets=2):
	"""
	Buckets by where it lies in the range max-min. 
	Ie if we had 1 dimensional data matrix [0,1,2,3,4,10] -> [0,0,0,0,1,2] for 3 buckets
	"""
	M = M.copy()
	buckets = []
	for i in range(0,num_buckets):
		temp = []
		for j in range(0,M.shape[1]):
			temp.append((M.T[j].max() - M.T[j].min())*i/(1.0 * num_buckets) + M.T[j].min())
		buckets.append(temp)
	for i in range(0,M.shape[0]):
		for j in range(0,M.shape[1]):
			max_bucket = 0
			for k in range(0,num_buckets):
				if M[i][j] >= buckets[k][j]:
					max_bucket = k
			M[i][j] = max_bucket
	return M
def bucketByMean(M):
	"""
	Binary bucketing scheme, sends value to 0 if < mean, else to 1
	Ie if we had 1 dimensional data matrix [0,1,2,3,4,10] -> [0,0,0,0,1,1] for 3 buckets
	"""
	converted_matrix = np.empty([M.shape[0], M.shape[1]], dtype=int)
	buckets = []
	num_buckets = 2
	buckets.append(np.zeros(M.shape[1]))
	for i in range(1,num_buckets):
		buckets.append(np.mean(M, axis=0))
	for i in range(0,M.shape[0]):
		for j in range(0,M.shape[1]):
			max_bucket = 0
			for k in range(0,num_buckets):
				if M[i][j] >= buckets[k][j]:
					max_bucket = k
			converted_matrix[i][j] = int(max_bucket)
	return converted_matrix

def IPF(graph, marginals, num_buckets, probability_update_function, num_iterations=100):
	"""
	graph: networkx graph
	marginals: Dictionary mapping edges -> np.array
	"""
	compatability_functions = {}
	for edge in graph.edges():
		compatability_functions[edge] = np.ones((num_buckets, num_buckets))
	probability_shape = []
	for _ in range(0, graph.number_of_nodes()):
		probability_shape += [num_buckets]
	probability_shape = tuple(probability_shape)
	probability = np.ones(probability_shape)
	probability = probability/np.sum(probability)
	axis = tuple(range(0,graph.number_of_nodes()))
	for _ in range(0,num_iterations):
		for edge in graph.edges():
			sum_axis = tuple(set(axis) - set(edge))
			compatability_functions[edge] = compatability_functions[edge] * marginals[edge]/np.sum(probability, axis=sum_axis)
			probability = probability_update_function(compatability_functions, num_buckets)
	return compatability_functions, probability

def strong_graph_update(compatability_functions,  num_buckets):
	probability = np.ones((num_buckets, num_buckets, num_buckets, num_buckets, num_buckets, num_buckets))
	for i in range(0, num_buckets):
		for j in range(0, num_buckets):
			for k in range(0, num_buckets):
				for l in range(0, num_buckets):
					for m in range(0, num_buckets):
						for n in range(0, num_buckets):
							probability[i][j][k][l][m][n] = compatability_functions[(0,1)][i][j] * compatability_functions[(0,2)][i][k] * compatability_functions[(0,5)][i][n]* compatability_functions[(1,3)][j][l] * compatability_functions[(1,4)][j][m] * compatability_functions[(2,4)][k][m] * compatability_functions[(3,5)][l][n] * compatability_functions[(4,5)][m][n]
	return probability/np.sum(probability)

def get_marginals_strong_graph(graph, data_matrix, num_buckets):
	marginals_shape = []
	for _ in range(0, graph.number_of_nodes()):
		marginals_shape += [num_buckets]
	marginals_shape = tuple(marginals_shape)
	marginals = np.zeros(marginals_shape)
	for sample in data_matrix:
		marginals[sample[0]][sample[1]][sample[2]][sample[3]][sample[4]][sample[5]] += 1
	marginals = marginals/np.sum(marginals)
	edge_marginals = {}
	for edge in graph.edges():
		axis = set([0,1,2,3,4,5])
		axis = tuple(axis - set(edge))
		edge_marginals[edge] = np.sum(marginals,axis=axis)
	return edge_marginals



tpm_bulk = cleanMatrix(tpm_bulk)
mean_bucketing = bucketByMean(tpm_bulk)
mean_bucketing = mean_bucketing.T
mean_bucketing = np.array([mean_bucketing[0], mean_bucketing[4], mean_bucketing[5], mean_bucketing[6], mean_bucketing[8], mean_bucketing[10]])
mean_bucketing = mean_bucketing.T
edge_marginals = get_marginals_strong_graph(strong_network, mean_bucketing, 2)
result = IPF(strong_network, edge_marginals, 2, strong_graph_update, 100)
