import numpy as np
import sys
import pandas as pd
import scipy
from sklearn.cluster import KMeans
import networkx as nx
import copy, random
import matplotlib.pyplot as plt
from SERGIO.SERGIO.sergio import sergio
from sklearn.preprocessing import StandardScaler



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
    #Add outlier genes (skipping)
#    expr_O = sim.outlier_effect(expr, outlier_prob = 0.01, mean = 0.8, scale = 1)
    expr_O = expr
    #Add Library Size Effect (skipping)
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



def get_DS_graph_MR(args): # 3 NOTE: please change the paths accordingly
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


def helper_GRN_data(args): # 2
    # initialize a random DAG
    if args.gen_DS in ['DS1', 'DS2', 'DS3']:
        # Use the given graph and simulate the data again
        G1, master_regulators = get_DS_graph_MR(args)            
    else:
        G1, master_regulators = random_DAG_with_MR(args)
    
    # saving the files, random number helps avoid clash 
    FILE_NUM = str(np.random.randint(1000))
    create_interaction_regs_files(G1, args, RANDOM_NUM=FILE_NUM)
    sim_data = get_data_SERGIO_batch(args, RANDOM_NUM=FILE_NUM, num_batch=1)
    return sim_data[0] + [master_regulators]

def create_GRN_data(args): # 1
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


def get_directed_graph(Gu):# 5
    Gd = nx.DiGraph()
    Gd.add_nodes_from(Gu.nodes)
    edges = Gu.edges
    Gd.add_edges_from(edges)
    return Gd
    
def random_DAG_with_MR(args): # 4
    """Generate a random Directed Acyclic Graph (DAG) with a given number of MR and sparsity."""
    prob = args.sparsity
    num_MR = int(prob * args.D)
    print('num MR = ', num_MR)
    
    master_regulators = np.array([n for n in range(num_MR)])
    other_nodes = np.array([num_MR + n for n in range(args.D-num_MR)])
    # Initializing a Bipartite graph
    G = nx.bipartite.random_graph(num_MR, args.D-num_MR, p=prob, seed=None, directed=False)
    # add minimal number of edges to make the graph connected
    edges = np.array([[e[0], e[1]] for e in G.edges])
    if len(edges) != 0:
        unconnected_MR = list(set(master_regulators) - set(edges[:, 0]))
        unconnected_ON = list(set(other_nodes) - set(edges[:, 1]))# other nodes
    else:
        unconnected_MR = list(set(master_regulators)) 
        unconnected_ON = list(set(other_nodes)) #other nodes
        
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
    unconnected_MR = list(set(master_regulators) - set(edges[:, 0]))
    unconnected_ON = list(set(other_nodes) - set(edges[:, 1]))# other nodes
    new_edges = []
    
    # make sure that each other node is connected
    for n in unconnected_ON:
        index = np.random.choice(len(master_regulators), 1, replace=False)
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
        for n in MR_A:
            index = np.random.choice(len(MR_B), 1, replace=False)
            new_edges.append([n, MR_B[index[0]]])
    # add the new edges
    G.add_edges_from(new_edges)
    # convert G to directed
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

def create_interaction_regs_files(G1, args, RANDOM_NUM=''):# 6 : NOTE: change the folder paths 
    # get master regulators: all nodes with in-degree zero
    node_in_degree = list(G1.in_degree(G1.nodes()))
    #node_degree = sorted(node_degree, key=lambda tup:tup[1])
    print('Master Regulators for regs file have 0 in degree: inferring using topological graph')
    master_regulators = np.array([n for n, d in node_in_degree if d==0])
    num_MR = len(master_regulators)
    # 1. edge list
    df_edge = pd.DataFrame(np.array(G1.edges()))
    df_edge.to_csv('simulator/SERGIO/data_sets/custom/gt_GRN'+RANDOM_NUM+'.csv', header=None, index=None)
    # 2. saving master regulator files


    # Prod_cell_rate = ~U(low_exp_range) & ~U(high_exp_range)
    low_cell_rate = np.random.rand(num_MR, args.C) * (args.pcr_low_max - args.pcr_low_min) + args.pcr_low_min  
    high_cell_rate = np.random.rand(num_MR, args.C) * (args.pcr_high_max - args.pcr_high_min) + args.pcr_high_min   
    mask = np.random.choice([0, 1], (num_MR, args.C))
    production_cell_rates = mask * low_cell_rate + (1 - mask) * high_cell_rate 

    master_reg_data = np.concatenate((master_regulators.reshape(num_MR, 1), production_cell_rates), 1)

    df_MR = pd.DataFrame(master_reg_data)
    df_MR.to_csv('simulator/SERGIO/data_sets/custom/Regs_cID'+RANDOM_NUM+'.txt', header=None, index=None)

    # 3. interaction file
    interaction_filename = 'simulator/SERGIO/data_sets/custom/Interaction_cID'+RANDOM_NUM+'.txt'
    interaction_nodes = np.array(list(G1.nodes()- master_regulators))
    #degree_dict = dict(G1.degree(interaction_nodes))
    incoming_nodes = [[v2, v1]for v1, v2 in np.array(G1.edges())]
    incoming_nodes_dict = {n:[] for n in G1.nodes()}
    incoming_degree_dict = {n:0 for n in G1.nodes()}
    for (n1, n2) in np.array(G1.edges()):
        incoming_nodes_dict[n2].append(n1)
        incoming_degree_dict[n2] += 1

    f = open(interaction_filename, 'w')
    for n in interaction_nodes:
        # NOTE: SERGIO SETTINGS CHANGED
        K_array = np.random.rand(len(incoming_nodes_dict[n])) * args.Kij_max - args.Kij_min + args.Kij_min #*4+5# * 4+1 #~U[1, 5]
        #K_array = K_array * np.random.choice([-1, 1], len(K_array)) # random +1/-1
        line = [n, incoming_degree_dict[n], *incoming_nodes_dict[n], *K_array]
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



def get_data_SERGIO_batch(args, u=1, RANDOM_NUM='', num_batch=1, PERMUTE=False):# 7
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
        sim.build_graph(input_file_taregts =folderpath+'Interaction_cID'+RANDOM_NUM+'.txt', input_file_regs=folderpath+'Regs_cID'+RANDOM_NUM+'.txt', shared_coop_state=args.SHARED_COOP_STATE)
        sim.simulate()
        expr = sim.getExpressions() # C x D x pts*B
    else:
        print('Check DATA_NAME')
    # get the labels
    if args.DATA_TYPE=='noisy':
        print('Adding noise ***')
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
    
    sim_data = []
    for b in range(num_batch):
        Xb = expr[:, :, b*args.POINTS_PER_CLASS:(b+1)*args.POINTS_PER_CLASS]
        Xb = np.concatenate(Xb, axis = 1).transpose()
        #print('Xb :', Xb.shape, expr.shape)
        yb = np.array([np.int(np.float(c/(args.POINTS_PER_CLASS))) for c in range(Xb.shape[0])])
        print('set labels check: ', set(yb), '\n Normalizing data')
        if PERMUTE: # permute Xb, the yb remains the same
            Xb = np.random.permutation(Xb.T).T
        sim_data.append([Xb, yb, theta_connections])

    return sim_data#edge_connections, theta_connections, X, y



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

def main():
    X, y = gen_data(100, 10, 5, 'random')
    print(X, y, X.shape, y.shape)
    return 

if __name__=="__main__":
    main()
