import numpy as np
import sys
import pandas as pd
import scipy
from sklearn.cluster import KMeans
import networkx as nx
#print(nx.__version__)
#if nx.__version__ != '1.11':
#    print('Wrong NETWORKx version!!! Will lead to wrong results')
import copy, random
import matplotlib.pyplot as plt
from SERGIO.SERGIO.sergio import sergio
from sklearn.preprocessing import StandardScaler


def gen_data(args, u=1):
    if args.DATA_METHOD == 'sim_expt1':
        DATA_NAME = 'DS1'
        TOTAL_SIMULATIONS = args.TOTAL_SIMULATIONS
        train_data = []
        for k in range(TOTAL_SIMULATIONS):
            print('simulation number = ', k, args.DATA_METHOD)
            edge_connections, theta_connections, X, y = get_data_SERGIO_saved(args, k)
#            Xc =  centering_kmeans(X, args.C)
#            Xc = get_exact_centering(X, args)
#            X = normalizing_data(X)
            #Xc = normalizing_data(Xc)
            Xc = normalizing_data(X)
            train_data.append([X, Xc, y, theta_connections])
            #print('NOTE: xc changed *******')
            #train_data.append([X, X, y, theta_connections])
        return train_data, [], []
    elif args.DATA_METHOD == 'sim_expt2':
        # get training/valid/test data
        train_data, valid_data, test_data = create_data_SERGIO(args)#, num_batch=args.K_train)
        return train_data, valid_data, test_data

    elif args.DATA_METHOD == 'sim_expt3':# Only the GRN reconstruction.
        if args.LOAD_DATA:
            print('Loading files from the data folder')
            train_data, valid_data, test_data = load_GRN_data(args)
        else: # create the data
            print('Generating the data')
            train_data, valid_data, test_data = create_GRN_data(args)
        return train_data, valid_data, test_data
        
    elif args.DATA_METHOD=='ds_expts':
        print('SEPARATELY implemented')
#        valid_data = create_data_SERGIO(args, num_batch=args.K_valid)
#        test_data = create_data_SERGIO(args, num_batch=args.K_test)
#            Xc =  centering_kmeans(X, args.C)
#            Xc = get_exact_centering(X, args)
#            X = normalizing_data(X)
#            Xc = normalizing_data(X)

        return 'implemented separately' #train_data, valid_data, test_data
        
    else: # synthetic experiments 
        return(prepare_synthetic_data(args))


# NOTE: this function is called from the main file
def get_DS_data(args):
    # load the data: train/valid/test = 5/5/5
    train_data, valid_data, test_data = [], [], []
    for k in range(args.K_train):
        print('train num = ', k)
        train_data.append(helper_DS_data(args, k, args.DATA_NAME))
    for _ in range(args.K_valid):
        k += 1
        print('valid num = ', k)
        valid_data.append(helper_DS_data(args, k, args.DATA_NAME))
    for _ in range(args.K_test):
        k += 1
        print('test num = ', k)
        test_data.append(helper_DS_data(args, k, args.DATA_NAME))
    return train_data, valid_data, test_data

# NOTE: this function is called from the main file
def get_DS_data_v2(args):
    # load the data: test = 15
    ds_data = {}
    for i, name in enumerate(['DS1', 'DS2', 'DS3']):
        ds_data[i] = [] 
        for k in range(15):# range(args.K_test)
#            print('test num = ', k)
            ds_data[i].append(helper_DS_data(args, k, name))
    return ds_data



def helper_DS_data(args, k, DATA_NAME, u=1):
    # get the master regulators
    if DATA_NAME == 'DS1':
        filepath = 'simulator/SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/'
        dim = 100
    elif DATA_NAME == 'DS2':
        filepath = 'simulator/SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/'
        dim = 400
    elif DATA_NAME == 'DS3':
        filepath = 'simulator/SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/'
        dim = 1200
    else:
        print('CHECK DATA NAME')
    given_edges = pd.read_csv(filepath+'gt_GRN.csv', header=None)
    given_edges = np.array(given_edges)
    master_regulators = set(given_edges[:, 0])
    if k==0:
        print(DATA_NAME, 'Num MR = ', len(master_regulators))
   
    # NOTE: something is wrong. 
 
    # get true theta
    G = get_graph(dim, given_edges)#, 'true_'+DATA_NAME)
    edge_connections = nx.adj_matrix(G).todense()
    smallest_eigval = np.min(np.linalg.eigvals(edge_connections))
    # precision matrix corresponding to edge_connections
    theta_connections = edge_connections + np.eye(dim)*(u- smallest_eigval)
    # load the data 
    sim_clean_exp = pd.read_csv(filepath+'simulated_noNoise_'+str(k)+'.csv', index_col=0)

    #sim = sergio(number_genes=args.D, number_bins = 9, number_sc = 300, noise_params = 1, decays=0.8, sampling_state=15, noise_type='dpd')
    sim_clean_exp = np.array(sim_clean_exp) 
    X = np.array(sim_clean_exp).transpose()# M x D = 2700 x 100
    # get the labels
    #y = np.array([np.int(np.float(c/args.POINTS)) for c in range(X.shape[0])])
    y = np.array([np.int(np.float(c/args.POINTS_PER_CLASS)) for c in range(X.shape[0])])
#    print('set labels check: ', set(y))
    return [X, y, theta_connections, list(master_regulators)]


# NOTE: this function is called from the main file
def add_technical_noise(args, data, name=0):
    print('Adding technical noise')
    # NOTE: Do no call sim.simulate()
    if args.DATA_METHOD == 'ds_expts' and name>0:
        dim_dict = {'DS1':100, 'DS2':400, 'DS3':1200}
#        dim = dim_dict[args.DATA_NAME]
        dim = dim_dict['DS'+str(name)]
        sim = sergio(number_genes=dim, number_bins = 9, number_sc = 300, noise_params = 1, decays=0.8, sampling_state=15, noise_type='dpd')
    else:
        sim = sergio(number_genes=args.D, number_bins = args.C, number_sc = args.POINTS_PER_CLASS, noise_params = args.NOISE_PARAMS, decays=args.DECAYS, sampling_state=args.SAMPLING_STATE, noise_type=args.NOISE_TYPE)
    noisy_data = []
    for i, d in enumerate(data):
        X, y, theta, MR = d
        X = helper_technical_noise(args, X, sim, name)
        noisy_data.append([X, y, theta, MR])
    return noisy_data

def helper_technical_noise(args, X, sim, name=0):# name =0 is default and runs for the input args setting: Use for training
    #print('clean shape: ', X.shape)
    expr = reshaping_sim_data(X.transpose(), args).transpose()
    #NOTE: Add outlier genes (skipping)
#    expr_O = sim.outlier_effect(expr, outlier_prob = 0.01, mean = 0.8, scale = 1)
    expr_O = expr
    #NOTE: Add Library Size Effect (skipping)
#    libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean = 4.6, scale = 0.4)
    expr_O_L = expr_O
    #Add Dropouts
    #binary_ind = sim.dropout_indicator(expr_O_L, shape = 6.5, percentile = 82)
    # more shape; less dropout --- lower percentile : less dropout
    if args.DATA_METHOD == 'ds_expts' and name>0:
        shape_dict = {'DS1':6.5, 'DS2':6.5, 'DS3':20}
        dropout_shape = shape_dict['DS'+str(name)]
        binary_ind = sim.dropout_indicator(expr_O_L, shape = dropout_shape, percentile = 82.0)
    else:
        binary_ind = sim.dropout_indicator(expr_O_L, shape = args.dropout_shape, percentile = args.dropout_percentile)
    print('BINARY IND: higher sum, less dropout ', binary_ind.size, np.sum(binary_ind), ' success rate = ', np.sum(binary_ind)/binary_ind.size)
    expr_O_L_D = np.multiply(binary_ind, expr_O_L)
    #Convert to UMI count
#    count_matrix = sim.convert_to_UMIcounts(expr_O_L_D)
    count_matrix = expr_O_L_D
    noisy_matrix = np.concatenate(count_matrix, axis = 1)
#    print('Noisy mat: ', noisy_matrix.shape)
    return noisy_matrix.transpose()



def get_DS_graph_MR(args):
    DATA_NAME = args.gen_DS
    if DATA_NAME == 'DS1':
        filepath = 'simulator/SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/'
        dim = 100
    elif DATA_NAME == 'DS2':
        filepath = 'simulator/SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/'
        dim = 400
    elif DATA_NAME == 'DS3':
        filepath = 'simulator/SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/'
        dim = 1200
    else:
        print('CHECK DATA NAME')
    given_edges = pd.read_csv(filepath+'gt_GRN.csv', header=None)
    given_edges = np.array(given_edges)
    master_regulators = list(set(given_edges[:, 0]))
    G = get_graph(args, given_edges)#, 'true_'+DATA_NAME)
    G = get_directed_graph(G)
    return G, master_regulators 


def helper_GRN_data(args):
    # initialize a random DAG
#    prob = args.sparsity
#    nodes = args.D
#    edges = nodes * (nodes-1)/2 * prob
#    G1 = random_dag(nodes, edges)
    if args.gen_DS in ['DS1', 'DS2', 'DS3']:
        # Use the given graph and simulate the data again
        G1, master_regulators = get_DS_graph_MR(args)            
    else:
        G1, master_regulators = random_DAG_with_MR(args)
    
    # saving the files, random number helps avoid clash 
    FILE_NUM = str(np.random.randint(1000))
#    master_regulators = create_interaction_regs_files(G1, args, RANDOM_NUM=FILE_NUM, master_regulators=master_regulators)
    create_interaction_regs_files(G1, args, RANDOM_NUM=FILE_NUM)
    sim_data = get_data_SERGIO_batch(args, RANDOM_NUM=FILE_NUM, num_batch=1)
    #print('sim data: ', sim_data[0])
    return sim_data[0] + [master_regulators]

def create_GRN_data(args):
    # create GRN data from the SERGIO simulator
    train_data, valid_data, test_data = [], [], []
    for k in range(args.K_train):
        print('train num = ', k)
        train_data.append(helper_GRN_data(args))
    for k in range(args.K_valid):
        print('valid num = ', k)
        valid_data.append(helper_GRN_data(args))
    for k in range(args.K_test):
        print('test num = ', k)
        test_data.append(helper_GRN_data(args))
    return train_data, valid_data, test_data


def get_directed_graph(Gu):
    Gd = nx.DiGraph()
    Gd.add_nodes_from(Gu.nodes)
    edges = Gu.edges
    Gd.add_edges_from(edges)
    return Gd
    
def random_DAG_with_MR(args):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of MR and sparsity."""
    prob = args.sparsity
#    nodes = args.D
#    edges = nodes * (nodes-1)/2 * prob
    num_MR = int(prob * args.D)
    print('num MR = ', num_MR)
    
    master_regulators = np.array([n for n in range(num_MR)])
    other_nodes = np.array([num_MR + n for n in range(args.D-num_MR)])
#    print(master_regulators, other_nodes)
    # Initializing a Bipartite graph
    G = nx.bipartite.random_graph(num_MR, args.D-num_MR, p=prob, seed=None, directed=False)
    # add minimal number of edges to make the graph connected
    edges = np.array([[e[0], e[1]] for e in G.edges])
#    print(edges)
    if len(edges) != 0:
        unconnected_MR = list(set(master_regulators) - set(edges[:, 0]))
        unconnected_ON = list(set(other_nodes) - set(edges[:, 1]))# other nodes
    else:
        unconnected_MR = list(set(master_regulators)) 
        unconnected_ON = list(set(other_nodes)) #other nodes
        
#    print('unconnected: ', unconnected_MR, unconnected_ON)
    # make sure that each MR as >= 1 out degree
    new_edges = []
    for n in unconnected_MR:
        # randomly select an edge from other nodes
        if len(unconnected_ON) > 0:
            index = np.random.choice(len(unconnected_ON), 1, replace=False)
#            print('unconnected index: ', index)
            new_edges.append([n, unconnected_ON[index[0]]])
        else:
            index = np.random.choice(len(other_nodes), 1, replace=False)
#            print('other nodes index: ', index)
            new_edges.append([n, other_nodes[index[0]]])
    # add the new edges
    G.add_edges_from(new_edges)
    
    # update arrays
    edges = np.array([[e[0], e[1]] for e in G.edges])
#    print('MR connected: ', edges)
    unconnected_MR = list(set(master_regulators) - set(edges[:, 0]))
#    print('CHECK: should be zero ', len(unconnected_MR))
    unconnected_ON = list(set(other_nodes) - set(edges[:, 1]))# other nodes
    new_edges = []
    
    # make sure that each other node is connected
    for n in unconnected_ON:
        index = np.random.choice(len(master_regulators), 1, replace=False)
#        print('other node index: ', index)
        new_edges.append([master_regulators[index[0]], n])
    
    # add the new edges
    G.add_edges_from(new_edges)
    # checking that each node has atleast one connection. 
    print('Final check: is DAG connected ?', set(np.array([[e[0], e[1]] for e in G.edges]).reshape(-1)) == set(range(args.D)))

    # Sanity check:
    edges = np.array([[e[0], e[1]] for e in get_directed_graph(G).edges])
    if edges[:, 0].all() != np.array(master_regulators).all():
        print('master regulators not matching', edges[:, 0], master_regulators)
    if len(master_regulators) > 1:
        num_connect_tf = int(args.connect_TF_prob * len(master_regulators))
        # select random pairs and join edges
        index = np.random.choice(len(master_regulators), num_connect_tf, replace=False)
        MR_A = set(master_regulators[index])
        MR_B = list(set(master_regulators) - MR_A)
        new_edges = []
#        print('MR_A = ', MR_A, MR_B)
        for n in MR_A:
            index = np.random.choice(len(MR_B), 1, replace=False)
            new_edges.append([n, MR_B[index[0]]])
    # add the new edges
#    print('new edges between tf:', new_edges)
    G.add_edges_from(new_edges)
#    print('all edges in undirected: ', G.edges, len(G.edges))
    # convert G to directed
    #G = G.to_directed()
    G = get_directed_graph(G)
    print('total edges = ', len(G.edges))
    return G, master_regulators



def load_saved_data(args):
    data_path = ''
    return train_data, valid_data, test_data


def normalizing_data(X):
    print('Normalising the input data...')
#    scaler = StandardScaler()
#    scaler.fit(X)  
#    scaledX = scaler.transform(X)

    scaledX = X - X.mean(axis=0)
    scaledX = scaledX/X.std(axis=0)
    # NOTE: replacing all nan's by 0, as sometimes in dropout the complete column
    # goes to zero
    scaledX = convert_nans_to_zeros(scaledX)
    return scaledX

def convert_nans_to_zeros(X):
    where_are_nans = isnan(X)
    X[where_are_nans] = 0
    return X


def get_exact_centering(X, args):
    Xc = []
    print('X shapei for centering: ', X.shape)
    for c in range(args.C):
        if args.DATA_METHOD == 'sim_expt1':
            data = np.array(X[c*args.DS1_POINTS:(c+1)*args.DS1_POINTS])
        elif args.DATA_METHOD == 'sim_expt2':
            data = np.array(X[c*args.POINTS_PER_CLASS:(c+1)*args.POINTS_PER_CLASS])
        print('Class = ', c, ' Num of points = ', len(data), data.shape, data.mean())
        #print('Class = ', l, ' Num of points = ', len(data), data.shape, data.mean(axis=0), data.mean(), centroids)
        Xc.append(data - data.mean(axis=0))
    Xc = np.concatenate(Xc, axis=0)
    print('temp Xc :', Xc.shape)
    Xc = Xc.reshape(-1, Xc.shape[-1])
    print('Exact centering result: ', Xc.shape)
    return Xc



def random_dag(nodes, edges):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    while edges > 0:
        a = random.randint(0,nodes-1)
        b=a
        while b==a:
            b = random.randint(0,nodes-1)
        G.add_edge(a,b)
        if nx.is_directed_acyclic_graph(G):
            edges -= 1
        else:
            # we closed a loop!
            G.remove_edge(a,b)

    # Check whether the graph is fully connected
    G_copy = G.copy()
    G_copy = G_copy.to_undirected()
    print('is DAG connected ? ', nx.is_connected(G_copy))
    # number of connected components
#    print('Connected components: ', nx.connected_components(G_copy))
    
    connected_components_nodes = sorted(nx.connected_components(G_copy), key=len, reverse=False)
    if len(connected_components_nodes) == 1:
        print('DAG is connected')
    else:
        print('Connect en edge with a node in smallest component to all the other components')
        n0 = list(connected_components_nodes[0])[0]
        print('cc: ', connected_components_nodes, n0)
        new_edges = []
        for c in connected_components_nodes[1:]:
            print(n0, c)
            new_edges.append([n0, list(c)[0]])
        # adding these new edges to G
        print('new edges = ', new_edges)
        G.add_edges_from(new_edges)
        
    print('Final check: is DAG connected ?', nx.is_connected(G.to_undirected()))
    #for i, c in enumerate(sorted(nx.connected_components(G_copy), key=len, reverse=False)):
    #    print('nodes', c)
        
    return G

def create_interaction_regs_files(G1, args, RANDOM_NUM=''):
    # get master regulators: all nodes with in-degree zero
    node_in_degree = list(G1.in_degree(G1.nodes()))
    #node_degree = sorted(node_degree, key=lambda tup:tup[1])
    print('Master Regulators for regs file have 0 in degree: inferring using topological graph')
    master_regulators = np.array([n for n, d in node_in_degree if d==0])
    num_MR = len(master_regulators)
    ##########*******NOTE: complete this part **********
    # 1. edge list
    df_edge = pd.DataFrame(np.array(G1.edges()))
    df_edge.to_csv('simulator/SERGIO/data_sets/custom/gt_GRN'+RANDOM_NUM+'.csv', header=None, index=None)
    # 2. saving master regulator files


    # Prod_cell_rate = ~U(low_exp_range) & ~U(high_exp_range)
    low_cell_rate = np.random.rand(num_MR, args.C) * (args.pcr_low_max - args.pcr_low_min) + args.pcr_low_min  
    high_cell_rate = np.random.rand(num_MR, args.C) * (args.pcr_high_max - args.pcr_high_min) + args.pcr_high_min   
    mask = np.random.choice([0, 1], (num_MR, args.C))
    production_cell_rates = mask * low_cell_rate + (1 - mask) * high_cell_rate 

#    production_cell_rates = np.random.rand(num_MR, args.C) * 0.03 # between[0.2, 0.5]
#    production_cell_rates = np.random.rand(num_MR, args.C)*20 #* 0.3  #* 0.3 # between[0.2, 0.5]
#    production_cell_rates = np.random.rand(num_MR, args.C) * 3 # between[0.2, 0.5]
#    production_cell_rates = np.random.rand(num_MR, args.C) * 10#0.3 # between[0.2, 0.5]
    #production_cell_rates2 = np.random.rand(len(master_regulators), args_C) * 0.3 + 0.7 # between[0.2, 0.5]
    # NOTE: CHANGED SERGIO SETTINGS

#    shift_choices = [i/10.0 for i in range(100)]

#    shift_choices = [0.1, 0.2, 0.3, 0.4, 0.5]#,  2, 5, 10]
#    shift_choices = [1, 2, 3, 4, 5]#,  2, 5, 10]
#    shift_choices = [0.1, 0.2, 0.3, 0.4, 0.5,  2, 5, 10]
#    shift_choices = [0.2, 0.7, 2, 5, 10]
    #shift_choices = [0.2, 0.7]
#    shift_choices = [0.7, 0.9]
#    shift_choices = [2.0, 2.0]
#    shift_choices = [1, 1]

#    shift_array = np.random.choice(shift_choices, (num_MR, args.C))
#    master_reg_data = np.concatenate((master_regulators.reshape(num_MR, 1), production_cell_rates+shift_array), 1)
    master_reg_data = np.concatenate((master_regulators.reshape(num_MR, 1), production_cell_rates), 1)

#    print(master_reg_data)
    df_MR = pd.DataFrame(master_reg_data)
    df_MR.to_csv('simulator/SERGIO/data_sets/custom/Regs_cID'+RANDOM_NUM+'.txt', header=None, index=None)

    # 3. interaction file
    interaction_filename = 'simulator/SERGIO/data_sets/custom/Interaction_cID'+RANDOM_NUM+'.txt'
    interaction_nodes = np.array(list(G1.nodes()- master_regulators))
    #degree_dict = dict(G1.degree(interaction_nodes))
    incoming_nodes = [[v2, v1]for v1, v2 in np.array(G1.edges())]
    incoming_nodes_dict = {n:[] for n in G1.nodes()}
    incoming_degree_dict = {n:0 for n in G1.nodes()}
    #print('Incoming nodes check : ', incoming_nodes_dict)
    for (n1, n2) in np.array(G1.edges()):
        incoming_nodes_dict[n2].append(n1)
        incoming_degree_dict[n2] += 1
    #print('Incoming nodes check : ', incoming_nodes_dict, incoming_degree_dict)
    #print('node degree: ', degree_dict, incoming_nodes)
#    print(interaction_nodes)

    f = open(interaction_filename, 'w')
    for n in interaction_nodes:
        # NOTE: SERGIO SETTINGS CHANGED
        K_array = np.random.rand(len(incoming_nodes_dict[n])) * args.Kij_max - args.Kij_min + args.Kij_min #*4+5# * 4+1 #~U[1, 5]
        #K_array = np.random.rand(len(incoming_nodes_dict[n]))*4+1#*4+5# * 4+1 #~U[1, 5]
        #K_array = np.random.rand(len(incoming_nodes_dict[n])) * 4+1 #~U[1, 5]
        # NOTE: SERGIO SETTINGS CHANGED
        K_array = K_array * np.random.choice([-1, 1], len(K_array)) # random +1/-1
        line = [n, incoming_degree_dict[n], *incoming_nodes_dict[n], *K_array]
    #    print('line = ', line)
        #for d in range(degree_dict[n]):
        for item in line:
            f.write("%f," % item)
        for i in range(incoming_degree_dict[n]):
            if i == incoming_degree_dict[n] - 1:
                f.write("2" % item)# shared coop value: integer
            else:
                f.write("2," % item)
        f.write('\n')
    f.close()

    print('SERGIO parameters used to create data: ')
    print('num_MR = ', num_MR, 'production cell rates = ', production_cell_rates,  ' Interaction K_array = ', K_array)  
    return # master_regulators


def create_data_SERGIO(args, k=None, seed=None):#, num_batch=1):
    # Generate the input files for SERGIO to simulate
    # initialize a DiGraph
    prob = args.sparsity
    nodes = args.D
    edges = nodes * (nodes-1)/2 * prob
    G1 = random_dag(nodes, edges)
    # saving the files, random number helps avoid clash 
    FILE_NUM = str(np.random.randint(1000))
    master_regulators = create_interaction_regs_files(G1, args, RANDOM_NUM=FILE_NUM)
    # load saved files and simulate data from SERGIO : NOTE: permutation invariant models needed
#    PERMUTE=False #True
#    print('Permutation invariant models needed = PERMUTE', PERMUTE, ' if true: just see classification results of CoNN in test')
#    sim_train = get_data_SERGIO_batch(args, RANDOM_NUM=FILE_NUM, num_batch=args.K_train) 
#    sim_valid = get_data_SERGIO_batch(args, RANDOM_NUM=FILE_NUM, num_batch=args.K_valid) 
#    sim_test = get_data_SERGIO_batch(args, RANDOM_NUM=FILE_NUM, num_batch=args.K_test, PERMUTE=PERMUTE) 

    sim_data = get_data_SERGIO_batch(args, RANDOM_NUM=FILE_NUM, num_batch=args.K_train+args.K_valid+args.K_test) 
    # dividing the data in train/test/valid
    sim_train = sim_data[:args.K_train]
    sim_valid = sim_data[args.K_train: args.K_train + args.K_valid]
    sim_test = sim_data[args.K_train+args.K_valid:]
    return sim_train, sim_valid, sim_test #edge_connections, theta_connections, X, y    

def get_data_SERGIO_batch(args, u=1, RANDOM_NUM='', num_batch=1, PERMUTE=False):
    DATA_NAME = args.DATA_NAME
    if DATA_NAME=='CUSTOM':
        folderpath = 'simulator/SERGIO/data_sets/custom/'
        print('Random Num  = ', RANDOM_NUM)
        edge_filepath = folderpath+'gt_GRN'+RANDOM_NUM+'.csv'
    else:# load the graph of the DS1 DS2 DS3 
        print('Check DATA_NAME')
    given_edges = pd.read_csv(edge_filepath, header=None)
    given_edges = np.array(given_edges)
    G = get_graph(args.D, given_edges)#, 'true_'+DATA_NAME)
    edge_connections = nx.adj_matrix(G).todense()
    smallest_eigval = np.min(np.linalg.eigvals(edge_connections))
    # precision matrix corresponding to edge_connections
    theta_connections = edge_connections + np.eye(args.D)*(u- smallest_eigval)
    # load the data 
    if DATA_NAME=='CUSTOM':
        # initialize the sergio simulator
        sim = sergio(number_genes=args.D, number_bins = args.C, number_sc = args.POINTS_PER_CLASS * num_batch, noise_params = args.NOISE_PARAMS, decays=args.DECAYS, sampling_state=args.SAMPLING_STATE, noise_type=args.NOISE_TYPE)
        # simulate data
        # NOTE: SERGIO CHANGES
        sim.build_graph(input_file_taregts =folderpath+'Interaction_cID'+RANDOM_NUM+'.txt', input_file_regs=folderpath+'Regs_cID'+RANDOM_NUM+'.txt', shared_coop_state=args.SHARED_COOP_STATE)
        sim.simulate()
        expr = sim.getExpressions() # C x D x pts*B
#        sim_clean_exp = np.concatenate(expr, axis = 1)
    else:
        print('Check DATA_NAME')
#    sim_clean_exp = np.array(sim_clean_exp) 
#    X = np.array(sim_clean_exp).transpose()# (M*B) x D = 2700 x 100
    # get the labels
    if args.DATA_TYPE=='noisy':
        print('Adding noise ***')
#        expr = reshaping_sim_data(sim_clean_exp, args)
        #Add outlier genes
        expr_O = sim.outlier_effect(expr, outlier_prob = 0.01, mean = 0.8, scale = 1)
        #Add Library Size Effect
        libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean = 4.6, scale = 0.4)
        #Add Dropouts
        binary_ind = sim.dropout_indicator(expr_O_L, shape = 6.5, percentile = 82)
        expr_O_L_D = np.multiply(binary_ind, expr_O_L)
        #Convert to UMI count
        count_matrix = sim.convert_to_UMIcounts(expr_O_L_D)
        expr = count_matrix
        #Make a 2d gene expression matrix
#        count_matrix = np.concatenate(count_matrix, axis = 1)
        # setting X 
#        X = count_matrix.transpose() # M x D 
    
    sim_data = []
    for b in range(num_batch):
        Xb = expr[:, :, b*args.POINTS_PER_CLASS:(b+1)*args.POINTS_PER_CLASS]
        Xb = np.concatenate(Xb, axis = 1).transpose()
        #print('Xb :', Xb.shape, expr.shape)
        yb = np.array([np.int(np.float(c/(args.POINTS_PER_CLASS))) for c in range(Xb.shape[0])])
        print('set labels check: ', set(yb), '\n Normalizing data')
        if PERMUTE: # permute Xb, the yb remains the same
            Xb = np.random.permutation(Xb.T).T
        #Xc = normalizing_data(Xb)
        #sim_data.append([Xb, Xc, yb, theta_connections])
        sim_data.append([Xb, yb, theta_connections])

    return sim_data#edge_connections, theta_connections, X, y
#    return train_data, valid_data, test_data #sim_data#edge_connections, theta_connections, X, y


def get_data_SERGIO_saved(args, k, u=1, RANDOM_NUM=''):
    DATA_NAME = args.DATA_NAME
    if DATA_NAME=='DS1':
        edge_filepath = 'simulator/SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/gt_GRN.csv'
    elif DATA_NAME=='CUSTOM':
        folderpath = 'simulator/SERGIO/data_sets/custom/'
        print('Random Num  = ', RANDOM_NUM)
        edge_filepath = folderpath+'gt_GRN'+RANDOM_NUM+'.csv'
    else: 
        print('Check DATA_NAME')
    given_edges = pd.read_csv(edge_filepath, header=None)
    given_edges = np.array(given_edges)
#    print('check edges: ', given_edges, given_edges.dtype)
    G = get_graph(args.D, given_edges)#, 'true_'+DATA_NAME)
    edge_connections = nx.adj_matrix(G).todense()
#    for i, e in enumerate(edge_connections):
#        print(e)
#        if i >0:
#            break
#    print('CHECKK:  see the theta') 
    smallest_eigval = np.min(np.linalg.eigvals(edge_connections))
    # precision matrix corresponding to edge_connections
    theta_connections = edge_connections + np.eye(args.D)*(u- smallest_eigval)
    # load the data 
    if DATA_NAME=='DS1':
        print('Loading the saved data for DS1: ', k)
        sim_clean_exp = pd.read_csv('simulator/SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/simulated_noNoise_'+str(k)+'.csv', index_col=0)

#        print('RESIMULATING DS1 data *************')
        sim = sergio(number_genes=100, number_bins = 9, number_sc = args.DS1_POINTS, noise_params = 1, decays=0.8, sampling_state=15, noise_type='dpd')
        #sim_clean_exp = pd.read_csv('simulator/SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/simulated_noNoise_'+str(4)+'.csv', index_col=0)
#        sim.build_graph(input_file_taregts ='simulator/SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt', input_file_regs='simulator/SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt', shared_coop_state=2)
#        sim.simulate()
#        expr = sim.getExpressions()
#        sim_clean_exp = np.concatenate(expr, axis = 1)
    elif DATA_NAME=='CUSTOM':
#        sim_clean_exp = pd.read_csv('simulator/SERGIO/data_sets/custom/simulated_noNoise_'+str(k)+'.csv', index_col=0)
        # initialize the sergio simulator
        sim = sergio(number_genes=args.D, number_bins = args.C, number_sc = args.POINTS_PER_CLASS, noise_params = args.NOISE_PARAMS, decays=args.DECAYS, sampling_state=args.SAMPLING_STATE, noise_type=args.NOISE_TYPE)
        # simulate data
        # NOTE: SERGIO CHANGES
        sim.build_graph(input_file_taregts =folderpath+'Interaction_cID'+RANDOM_NUM+'.txt', input_file_regs=folderpath+'Regs_cID'+RANDOM_NUM+'.txt', shared_coop_state=args.SHARED_COOP_STATE)
        sim.simulate()
        expr = sim.getExpressions()
        sim_clean_exp = np.concatenate(expr, axis = 1)
    else:
        print('Check DATA_NAME')
    sim_clean_exp = np.array(sim_clean_exp) 
    X = np.array(sim_clean_exp).transpose()# M x D = 2700 x 100
    # get the labels
    if DATA_NAME =='DS1':
        y = np.array([np.int(np.float(c/args.DS1_POINTS)) for c in range(X.shape[0])])
    elif DATA_NAME =='CUSTOM':
        y = np.array([np.int(np.float(c/args.POINTS_PER_CLASS)) for c in range(X.shape[0])])
    print('set labels check: ', set(y))
    if args.DATA_TYPE=='noisy':
        print('Adding noise ***')
        expr = reshaping_sim_data(sim_clean_exp, args)
        #Add outlier genes
        expr_O = sim.outlier_effect(expr, outlier_prob = 0.01, mean = 0.8, scale = 1)
        #Add Library Size Effect
        libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean = 4.6, scale = 0.4)
        #Add Dropouts
        binary_ind = sim.dropout_indicator(expr_O_L, shape = 6.5, percentile = 82)
        expr_O_L_D = np.multiply(binary_ind, expr_O_L)
        #Convert to UMI count
        count_matrix = sim.convert_to_UMIcounts(expr_O_L_D)
        #Make a 2d gene expression matrix
        count_matrix = np.concatenate(count_matrix, axis = 1)
        # setting X 
        X = count_matrix.transpose() # M x D 
    return edge_connections, theta_connections, X, y

def reshaping_sim_data(sim_clean_exp, args):
    # reshaping, to add technical noise 9 x 100 x 300
    reshaped_sim_clean_exp = []
    for ct in range(args.C): # enumerate through the cell types
        reshaped_sim_clean_exp.append(sim_clean_exp[:, args.POINTS_PER_CLASS*ct:args.POINTS_PER_CLASS*(ct+1)])
    reshaped_sim_clean_exp = np.array(reshaped_sim_clean_exp)
    #print(reshaped_sim_clean_exp_0.shape)
    return reshaped_sim_clean_exp


def get_graph(D, edges):
    G=nx.Graph()#nx.DiGraph()
    G.add_nodes_from([n for n in range(D)])
    G.add_edges_from(edges)
    return G


def prepare_synthetic_data(args):
    train_data, valid_data, test_data = [], [], []
    edge_connections, theta_connections = get_sparsity_pattern(args) # the adjacency matrix
#        print(edge_connections, theta_connections) # the adjacency matrix
    for k in range(args.K_train):
        if args.SUB_METHOD=='case2':
            edge_connections, theta_connections = get_sparsity_pattern(args) # the adjacency matrix
        X, Xc, y = get_classwise_samples(args, edge_connections)
        train_data.append([X, Xc, y, theta_connections])


    for k in range(args.K_valid):
        if args.SUB_METHOD=='case2':
            edge_connections, theta_connections = get_sparsity_pattern(args) # the adjacency matrix
#            print('VALID: ', edge_connections, theta_connections) # the adjacency matrix
        X, Xc, y = get_classwise_samples(args, edge_connections, TEST=True)
        valid_data.append([X, Xc, y, theta_connections])

    for k in range(args.K_test):
        if args.SUB_METHOD=='case2':
            edge_connections, theta_connections = get_sparsity_pattern(args) # the adjacency matrix
#            print('TEST: ', edge_connections, theta_connections) # the adjacency matrix
        X, Xc, y = get_classwise_samples(args, edge_connections, seed=k, TEST=True)
        test_data.append([X, Xc, y, theta_connections])
    
    return train_data, valid_data, test_data


def get_sparsity_pattern(args, seed=None, u=1):
    if seed != None:
        np.random.seed(seed)
    prob = args.sparsity
    G = nx.generators.random_graphs.gnp_random_graph(args.D, prob, seed=seed, directed=False)
    edge_connections = nx.adjacency_matrix(G).todense() # adjacency matrix
    smallest_eigval = np.min(np.linalg.eigvals(edge_connections))
    # Just in case : to avoid numerical error in case a epsilon complex component present
    smallest_eigval = smallest_eigval.real
    # precision matrix corresponding to edge_connections
    theta_connections = edge_connections + np.eye(args.D)*(u- smallest_eigval)
#    print('******** DELETE LATER: SERGIO sparsity pattern')
#    edge_filepath = 'simulator/SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/gt_GRN.csv'
#    given_edges = pd.read_csv(edge_filepath, header=None)
#    given_edges = np.array(given_edges)
#    G = get_graph(given_edges)#, 'true_'+DATA_NAME)
#    edge_connections = nx.adj_matrix(G).todense()
#    smallest_eigval = np.min(np.linalg.eigvals(edge_connections))
#    print('edge_connections: smallest eval = ', smallest_eigval)
#    smallest_eigval = smallest_eigval.real
#    print('edge_connections after: smallest eval = ', smallest_eigval)
#    # precision matrix corresponding to edge_connections
#    theta_connections = edge_connections + np.eye(args.D)*(u- smallest_eigval)
    return edge_connections, theta_connections

def centering_kmeans(X, k):
    # center X using the centroids
    # running Kmeans on X using K cluster centers
#    print('Check X: ', X, X.shape)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
#    print('Labels : ', kmeans.labels_)
#    print('Centers: ', kmeans.cluster_centers_)
    Xc = []
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    for l in range(k):
        data = np.array(X[labels==l])#-centroids[l]
        print('Class = ', l, ' Num of points = ', len(data), data.shape, data.mean())
        #print('Class = ', l, ' Num of points = ', len(data), data.shape, data.mean(axis=0), data.mean(), centroids)
        Xc.append(data - data.mean(axis=0))
    Xc = np.concatenate(Xc, axis=0)
    print('Xc :', Xc.shape)
    Xc = Xc.reshape(-1, Xc.shape[-1])
    print('result: ', Xc.shape)
#    brr
    return Xc
    

def get_classwise_samples(args, edge_connections, seed=None, TEST=False, u=1):
    D, C, method, prob, mean_gap = args.D, args.C, args.DATA_METHOD, args.sparsity, args.mean_gap
    w_min, w_max = args.w_min, args.w_max
    if method=='syn_expt2': 
        description = 'mean depends on class, same edge connections with different precision values, do centering using K-means/DBscan' 
        X, y = [], [] # Xc= X_centered
        for c in range(C):
            mean_value = mean_gap * c 
            mean_normal = np.ones(D) * mean_value
            if seed != None:
                np.random.seed(seed+c)
            # uniform [w_min, w_max]
            U = np.matrix(np.random.random((D, D)) * (w_max - w_min) + w_min)
            theta = np.multiply(edge_connections, U)
            # making it symmetric
            theta = (theta + theta.T)/2 + np.eye(D)
            smallest_eigval = np.min(np.linalg.eigvals(theta))
            # Just in case : to avoid numerical error in case a epsilon complex component present
            smallest_eigval = smallest_eigval.real
            # making the min eigenvalue as 1
            precision_mat = theta + np.eye(D)*(u - smallest_eigval)
            print('CHEKKK: smallest eigen? = ', np.min(np.linalg.eigvals(precision_mat)))
            cov = np.linalg.inv(precision_mat) # avoiding the use of pinv as data not true representative of conditional independencies.
            # get the samples corresponding to the class c
            if TEST:
                sample_size = int(args.M_test/C) #int((1-args.alpha)*M/C)
            else:
                sample_size = int(args.M_train/C)#int(args.alpha*M/C)
            if seed != None:
                np.random.seed(seed+c)
            data = np.random.multivariate_normal(mean=mean_normal, cov=cov, size=sample_size) #.T
#            print('chhch: ', data.shape, data)
            X.append(data)
            y.append(np.ones(sample_size)*c)
#            Xc.append(data-data.mean(axis=0))
            print('True mean = ', mean_value)#, 'estimated mean = ', data.mean())#data.mean(axis=0))
        X = np.array(X)
        X = X.reshape(-1, X.shape[-1])
        #Xc = np.array(Xc)
        #Xc = Xc.reshape(-1, Xc.shape[-1])
        Xc = centering_kmeans(X, C)
        y = np.array(y).reshape(-1)
        return X, Xc, y # X = (MC)xD, Mx1

    elif method=='syn_expt1': 
        description = 'mean depends on class, same edge connections with different precision values' 
        X, y, Xc = [], [], [] # Xc= X_centered
        for c in range(C):
            mean_value = mean_gap * c 
            mean_normal = np.ones(D) * mean_value
            if seed != None:
                np.random.seed(seed+c)
            # uniform [w_min, w_max]
            U = np.matrix(np.random.random((D, D)) * (w_max - w_min) + w_min)
            theta = np.multiply(edge_connections, U)
            # making it symmetric
            theta = (theta + theta.T)/2 + np.eye(D)
            smallest_eigval = np.min(np.linalg.eigvals(theta))
            # making the min eigenvalue as 1
            precision_mat = theta + np.eye(D)*(u - smallest_eigval)
            print('CHEKKK: smallest eigen? = ', np.min(np.linalg.eigvals(precision_mat)))
            cov = np.linalg.inv(precision_mat) # avoiding the use of pinv as data not true representative of conditional independencies.
            # get the samples corresponding to the class c
            if TEST:
                sample_size = int(args.M_test/C) #int((1-args.alpha)*M/C)
            else:
                sample_size = int(args.M_train/C)#int(args.alpha*M/C)
            if seed != None:
                np.random.seed(seed+c)
            data = np.random.multivariate_normal(mean=mean_normal, cov=cov, size=sample_size) #.T
            X.append(data)
            y.append(np.ones(sample_size)*c)
            Xc.append(data-data.mean(axis=0))
            print('True mean = ', mean_value, 'estimated mean = ', data.mean())#data.mean(axis=0))
        X = np.array(X)
        X = X.reshape(-1, X.shape[-1])
        Xc = np.array(Xc)
        Xc = Xc.reshape(-1, Xc.shape[-1])
        y = np.array(y).reshape(-1)
        return X, Xc, y # X = (MC)xD, Mx1
    




def main():
    X, y = gen_data(100, 10, 5, 'random')
    print(X, y, X.shape, y.shape)
    return 

if __name__=="__main__":
    main()
