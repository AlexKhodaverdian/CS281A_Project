import networkx as nx
import numpy as np

"""
Constructing our networks:
1) strong_network is the network with only strong edges
2) regular_network is the network with both strong and weak edges
Note: The vertices in my network map back to original network,
ie for strong network [0,1,2,3,4,5] <---> [1,5,6,7,9,11]
   for the regular network [0,1,2,3,4,5,6,7,8] <---> [1,2,3,5,6,7,8,9,11]
"""
strong_network = nx.Graph()
regular_network = nx.Graph()

strong_network.add_edge(0,1)
strong_network.add_edge(0,2)
strong_network.add_edge(0,5)
strong_network.add_edge(1,3)
strong_network.add_edge(1,4)
strong_network.add_edge(2,4)
strong_network.add_edge(3,5)
strong_network.add_edge(4,5)

regular_network.add_edge(0,1)
regular_network.add_edge(0,2)
regular_network.add_edge(0,3)
regular_network.add_edge(0,4)
regular_network.add_edge(0,5)
regular_network.add_edge(0,6)
regular_network.add_edge(0,7)
regular_network.add_edge(0,8)
regular_network.add_edge(1,7)
regular_network.add_edge(2,5)
regular_network.add_edge(2,7)
regular_network.add_edge(3,5)
regular_network.add_edge(3,7)
regular_network.add_edge(4,5)
regular_network.add_edge(4,7)
regular_network.add_edge(5,6)
regular_network.add_edge(5,7)
regular_network.add_edge(5,8)
regular_network.add_edge(6,7)
regular_network.add_edge(7,8)

def cleanMatrix(M):
	"""
	M: Data Matrix
	Returns a new matrix with all samples(rows) with dropout (ie 0 in data matrix) removed
	"""
	new_matrix = []
	for sample in M:
		if not 0 in sample:
			new_matrix.append(sample)
	return np.array(new_matrix)

"""
Loading in the data and cleaning it
1) tpm_bulk is the bulk single cell data, where rows are the samples and columns are genes
2) tpm_single_cell is the single cell data, where rows are the samples and columns are genes
Note for both: I stripped out all columns that were not necessary:
ie for the strong network, I stripped out columns corresponding to nodes: 2,3,4,8,10,12
   for the regular network, I stripped out columns corresponding to nodes: 4,10,12
ALSO: I clean out all rows with dropout in the nodes we're interested in
"""
tpm_bulk=np.loadtxt('SRSF_bulk.tsv', delimiter=',', usecols=range(1,34))
tpm_single_cell=np.loadtxt('SRSF_scRNA.tsv', delimiter=',', usecols=range(1,2366))
tpm_bulk = np.array([tpm_bulk[0], tpm_bulk[7], tpm_bulk[8], tpm_bulk[9], tpm_bulk[11], tpm_bulk[2]])
tpm_single_cell = np.array([tpm_single_cell[0], tpm_single_cell[4], tpm_single_cell[5], tpm_single_cell[7], tpm_single_cell[8], tpm_single_cell[9], tpm_single_cell[10], tpm_single_cell[11], tpm_single_cell[2]])
tpm_bulk = tpm_bulk.T
tpm_single_cell = tpm_single_cell.T
tpm_bulk = cleanMatrix(tpm_bulk)
tpm_single_cell = cleanMatrix(tpm_single_cell)



def bucketByMedian(M, num_buckets=2):
	"""
	M: Data Matrix
	num_buckets: Number of buckets to partition M into
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
	M: Data Matrix
	num_buckets: Number of buckets to partition M into
	Buckets by where it lies in the range max-min. 
	Ie if we had 1 dimensional data matrix [0,1,2,3,4,10] -> [0,0,0,0,1,2] for 3 buckets (ie [3.333, 6.666])
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
	M: Data Matrix
	Binary bucketing scheme, sends value to 0 if < mean, else to 1
	Ie if we had 1 dimensional data matrix [0,1,2,3,4,10] -> [0,0,0,0,1,1] for 3 buckets (Mean = 3.33333)
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
	marginals: Dictionary mapping edges -> np.array that is the edge-wise marginals of our data
	num_buckets: The number of buckets used for the marginals
	probability_update_function: The function passed in to update the probabilities after every iteration of IPF
	num_iterations = Number of iterations to run IPF

	Runs IPF and returns a tuple containing a dictionary of compatability functions (Edge -> tsi) and the overall probability
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
			sum_of_probability = np.sum(probability, axis=sum_axis)
			sum_of_probability[sum_of_probability == 0] = 1
			compatability_functions[edge] = compatability_functions[edge] * marginals[edge]/sum_of_probability
			probability = probability_update_function(compatability_functions, num_buckets)
	return compatability_functions, probability

def strong_graph_update(compatability_functions,  num_buckets):
	"""
	compatability_functuons: A dictionary mapping edge -> tsi
	num_buckets: The number of buckets used when generating the marginals
	Given the compatability functions, returns the probability matrix for the strongly connected graph
	"""
	probability = np.ones((num_buckets, num_buckets, num_buckets, num_buckets, num_buckets, num_buckets))
	for i in range(0, num_buckets):
		for j in range(0, num_buckets):
			for k in range(0, num_buckets):
				for l in range(0, num_buckets):
					for m in range(0, num_buckets):
						for n in range(0, num_buckets):
							probability[i][j][k][l][m][n] = compatability_functions[(0,1)][i][j] * compatability_functions[(0,2)][i][k] * compatability_functions[(0,5)][i][n]* compatability_functions[(1,3)][j][l] * compatability_functions[(1,4)][j][m] * compatability_functions[(2,4)][k][m] * compatability_functions[(3,5)][l][n] * compatability_functions[(4,5)][m][n]
	return probability/np.sum(probability)

def regular_graph_update(compatability_functions,  num_buckets):
	"""
	compatability_functuons: A dictionary mapping edge -> tsi
	num_buckets: The number of buckets used when generating the marginals
	Given the compatability functions, returns the probability matrix for the regular connected graph
	"""
	probability = np.ones((num_buckets, num_buckets, num_buckets, num_buckets, num_buckets, num_buckets, num_buckets, num_buckets, num_buckets))
	for i in range(0, num_buckets):
		for j in range(0, num_buckets):
			for k in range(0, num_buckets):
				for l in range(0, num_buckets):
					for m in range(0, num_buckets):
						for n in range(0, num_buckets):
							for o in range(0, num_buckets):
								for p in range(0, num_buckets):
									for q in range(0, num_buckets):
										probability[i][j][k][l][m][n][o][p][q] = compatability_functions[(0,1)][i][j] * compatability_functions[(0,2)][i][k] * compatability_functions[(0,3)][i][l] * compatability_functions[(0,4)][i][m] *  compatability_functions[(0,5)][i][n]* compatability_functions[(0,6)][i][o] * compatability_functions[(0,7)][i][p] *  compatability_functions[(0,8)][i][q] 
										probability[i][j][k][l][m][n][o][p][q] = probability[i][j][k][l][m][n][o][p][q] * compatability_functions[(1,7)][j][p] * compatability_functions[(2,5)][k][n] * compatability_functions[(2,7)][k][p] * compatability_functions[(3,5)][l][n] * compatability_functions[(3,7)][l][p] * compatability_functions[(4,5)][m][n] * compatability_functions[(4,7)][m][p]
										probability[i][j][k][l][m][n][o][p][q] = probability[i][j][k][l][m][n][o][p][q]  * compatability_functions[(5,6)][n][o] * compatability_functions[(5,7)][n][p] * compatability_functions[(5,8)][n][q] * compatability_functions[(6,7)][o][p] * compatability_functions[(7,8)][p][q] 
	return probability/np.sum(probability)

def get_marginals_strong_graph(graph, data_matrix, num_buckets=2):
	"""
	graph: networkx graph (Input should be the graph corresponding to the strong_network)
	data_matrix: The matrix corresponding to the data AFTER bucketing(ie: tpm_bulk or tpm_single_cell)
	num_buckets: Number of buckets the data was divided into
	"""
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

def get_marginals_regular_graph(graph, data_matrix, num_buckets=2):
	"""
	graph: networkx graph (Input should be the graph corresponding to the regular_network)
	data_matrix: The matrix corresponding to the data AFTER bucketing(ie: tpm_bulk or tpm_single_cell)
	num_buckets: Number of buckets the data was divided into
	"""
	marginals_shape = []
	for _ in range(0, graph.number_of_nodes()):
		marginals_shape += [num_buckets]
	marginals_shape = tuple(marginals_shape)
	marginals = np.zeros(marginals_shape)
	for sample in data_matrix:
		marginals[sample[0]][sample[1]][sample[2]][sample[3]][sample[4]][sample[5]][sample[6]][sample[7]][sample[8]] += 1
	marginals = marginals/np.sum(marginals)
	edge_marginals = {}
	for edge in graph.edges():
		axis = set([0,1,2,3,4,5,6,7,8])
		axis = tuple(axis - set(edge))
		edge_marginals[edge] = np.sum(marginals,axis=axis)
	return edge_marginals


#mean_bucketing = bucketByMedian(tpm_bulk, num_buckets=3)
#edge_marginals_2 = get_marginals_strong_graph(strong_network, mean_bucketing, num_buckets=3)
#compatability_functions_bulk, probability_bulk = IPF(strong_network, edge_marginals_2, 3, strong_graph_update, 50)

#mean_bucketing = bucketByMedian(tpm_single_cell,num_buckets=2)
#edge_marginals = get_marginals_regular_graph(regular_network, mean_bucketing, 2)
#compatability_functions_single_cell, probability_single_cell = IPF(regular_network, edge_marginals, 2, regular_graph_update, 10)

bulkMedian = bucketByMedian(tpm_bulk)
bulkMM= get_marginals_strong_graph(strong_network,bulkMedian)
IPF(strong_network,bulkMM,2,strong_graph_update)
