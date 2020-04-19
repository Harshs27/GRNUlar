# We have not released the BEELINE data and ground truth network shared by the authors
# due to privacy considerations. We are anyway making the code used for preprocessing
# available. You will need to update the input folder and other basic I/O to get it 
# to work. Needless, the preprocessing steps followed will be clear by following the
# code. Please feel free to contact me (Harsh) via direct mail or raising a github 
# issue.

import numpy as np
import pandas as pd
print('pandas version: ', pd.__version__)
import networkx as nx
print(nx.__version__)
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
print('The scikit-learn version is {}.'.format(sklearn.__version__))
import operator, itertools, os, sys
from platform import python_version
import pickle
print(python_version())


def get_graph(edges, WEIGHTED=False):
    G = nx.DiGraph()
    #G.add_nodes_from(['G'+str(n) for n in range(100)])
    #print('check: ', edges, WEIGHTED)
    if WEIGHTED:
        G.add_weighted_edges_from(edges)
    else:
        G.add_edges_from(edges)
    #fig = plt.figure(figsize=(15, 15))
    #nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels = True)
    #nx.draw_networkx(G, with_labels = True)
    return G


def get_PSD_matrix(G, u=1):
    A = nx.adj_matrix(G).todense()
    smallest_eigval = np.min(np.linalg.eigvals(A))
    # Just in case : to avoid numerical error in case a epsilon complex component present
    smallest_eigval = smallest_eigval.real
    # making the min eigenvalue as 1
    target_precision_mat = A + np.eye(A.shape[-1])*(u - smallest_eigval)
    #print('CHEKKK: smallest eigen? = ', np.min(np.linalg.eigvals(target_precision_mat)))
    return target_precision_mat.real



def remove_inactive_genes(expr):
    # remove genes with < 10% activity
    expr_0 = expr.copy()
    expr_0_sub = expr_0[expr_0.columns[1:]]
    # converting all the non-zeros to 1
    expr_0_sub[expr_0_sub != 0] = 1
    expr_0[expr_0.columns[1:]] = expr_0_sub 
    #print(expr_0, np.array(expr_0))
    # summing the columns
    expr_0 = np.array(expr_0)
    total_expts = expr.shape[1]
    print('total_expts = ', total_expts)
    expr_0[:, 1] = np.sum(expr_0[:, 1:], 1)/total_expts
    expr_0 = pd.DataFrame(expr_0[:, 0:2])
    # dropping all genes with < 90%
    expr_0 = expr_0[expr_0[1] > 0.1]
    # get the active genes
    active_genes = expr_0[0]
    print('active genes', len(active_genes))
    return expr.loc[expr['Unnamed: 0'].isin(active_genes)]


def get_filename_network(network_name, data_name):
    if network_name in ['STRING-network', 'Non-Specific-ChIP-seq-network']:
        return network_name + '.csv'
    else:
        if '-' in data_name:
            data_name = data_name.split('-')[0]
        if data_name == 'hHep':
            data_name = 'HepG2'
        return data_name + '-ChIP-seq-network.csv'

    
def get_top_varying_genes(gene_ordering, genes_in_expr_network, TF_all, NUM=500):
    #We started by including all TFs whose variance had P value at most 0.01.
    #Then, we added 500 and 1,000 additional genes as in the previous option.
    #This approach enabled the GRN methods to consider TFs that may have a
    #modest variation in gene expression but still regulate their targets.
    
    #print('before: ', gene_ordering)
    # select the subset of genes with only expr_genes present 
    gene_ordering = gene_ordering[gene_ordering['Unnamed: 0'].isin(genes_in_expr_network)]
    #print('subset expr: ', gene_ordering)
    # consider all the genes with p-values <=0.01
    gene_ordering = gene_ordering[gene_ordering['VGAMpValue'] < 0.01]
    #print('p-val <= 0.01: ', gene_ordering)
    
    genes_in_ordering = gene_ordering['Unnamed: 0']
    genes_in_ordering = convert_to_captial(genes_in_ordering)
    print('genes with p<0.1 in gene ordering : ', len(genes_in_ordering))
    
    TF_pval = list(set(TF_all) & set(genes_in_ordering))
    print('set of TFs with p<0.1', len(TF_pval))
 
    gene_ordering = gene_ordering.sort_values(['Variance'], ascending=False)
    #print('sorted by variance: ', gene_ordering)
    
    top_varying_genes = np.array(gene_ordering['Unnamed: 0'])[:NUM]
    top_varying_genes = convert_to_captial(top_varying_genes)
    print('top_varying_genes', len(top_varying_genes))
    
    # find intersection with top varying genes
    TF_sub = list( set(top_varying_genes) & set(TF_pval))
    #TF_sub = list( set(top_varying_genes) & set(TF_all))

    print('**TF_sub', len(TF_sub))
    return top_varying_genes, TF_sub

def convert_to_captial(genes):
    return np.array([g.upper() for g in genes])
    
    
def get_real_data(network_name, data_name):
    data_name_dict = {}
    # load the expression data
    folder_expr = 'simulator/BEELINE-data/inputs/scRNA-Seq/'
    folder_network = 'simulator/BEELINE-data/Networks/'
    expr = pd.read_csv(folder_expr + data_name + '/ExpressionData.csv') # genes x expts
    expr_genes = np.array(expr['Unnamed: 0'])
    expr_genes = convert_to_captial(expr_genes)
    expr['Unnamed: 0'] = expr_genes
    print('expr genes', len(expr_genes))
    
    
    # removing the genes with more than 90% zeros
    expr = remove_inactive_genes(expr)
    # corresponding mDC network
    if data_name[0] == 'm':
        species = 'mouse'
    else:
        species = 'human'
    filename_network = get_filename_network(network_name, data_name)
    print('given edgdes: filename = ', folder_network + species + '/'+filename_network)
    given_edges = pd.read_csv(folder_network + species + '/'+filename_network)
    given_edges['Gene1'] = convert_to_captial(given_edges['Gene1'])
    given_edges['Gene2'] = convert_to_captial(given_edges['Gene2'])
    #print(given_edges)
    
    print('Given edges: all', len(given_edges))
    TF_from_network = list(set(np.array(given_edges['Gene1'])))
    print('TF_from_network', len(TF_from_network))
    
    weighted=False
    if 'Score' in given_edges.columns:
        weighted=True
    # get the true graph
    G_true = get_graph(np.array(given_edges), weighted)
    total_genes_network = G_true.nodes
    total_genes_network = convert_to_captial(total_genes_network)
    print('Total genes in network: ', len(total_genes_network))
   
    
    genes_in_expr_network = list(set(expr_genes) & set(total_genes_network))
    print('Common genes in network & expr: SHOULD be high', len(genes_in_expr_network))
    
    # get the TFs
    TF_all = np.array(pd.read_csv(folder_network + species + '-tfs.csv')['TF'])
    TF_all = convert_to_captial(TF_all)
    print('TF_all', len(TF_all))
    # get the top 500 most varying genes
    gene_ordering = pd.read_csv(folder_expr + data_name + '/GeneOrdering.csv')
    
    gene_ordering['Unnamed: 0'] = convert_to_captial(gene_ordering['Unnamed: 0'])
    #print(gene_ordering)
    
    top_varying_genes, TF_sub = get_top_varying_genes(gene_ordering, genes_in_expr_network, TF_all, NUM=500)
    
    # select the subset of TFs
    print('Sanity check 1: # of TF from network < TF all : ', len(TF_from_network), len(TF_all), 
          len(set(TF_from_network) & set(TF_all)))
    
    # select the subnetwork of TFs+500 genes from the G_true 
    sub_network_genes = list(set(top_varying_genes) | set(TF_sub)) # union
    G_sub = G_true.subgraph(sub_network_genes)
    
    print('Sanity check 2: Total genes in subnetwork : ', len(G_sub.nodes), len(sub_network_genes))
    
    G_sub_out_deg = pd.DataFrame(G_sub.out_degree(G_sub.nodes))
    #G_sub_out_deg = G_sub_out_deg.sort_values([1], ascending=False)
    G_sub_out_deg = G_sub_out_deg[G_sub_out_deg[1]>0]
    #print('check :', G_sub_out_deg )
    
    TF_sub_true = np.array(G_sub_out_deg[0])
    print('Sanity check 3: TF from true network <= TF_sub comparison', len(TF_sub_true), len(TF_sub), 
         len(set(TF_sub_true) & set(TF_sub)))
    #TF_sub1 = list(set(TF_all) & set(top_varying_genes) & set(TF_from_network))
    #print('TF_sub', len(TF_sub))
    
    #TF_sub = list(set(TF_all) & set(expr_genes) & set(top_varying_genes))
    #print('TF_sub', len(TF_sub))
    
    # get theta_true
    theta_true = get_PSD_matrix(G_sub)
    # expression data of the subset of genes, after pre-processing
    X = expr.loc[expr['Unnamed: 0'].isin(np.array(G_sub.nodes))]
    X.index = X['Unnamed: 0']
    X = X[X.columns[1:]]
    # SYNC between X and theta
    #print('SYNC: ',X.loc['SOD2'], X.index, G_sub.nodes)
    X = X.reindex(G_sub.nodes)
    #print('After: ',X.loc['SOD2'], X.index, G_sub.nodes)
    
    X = X.transpose()
    print('X = expts x genes: Theta shape: ', X.shape, theta_true.shape)
    
    print('Convert the genes to numbers')
    #print('X check', X.columns)
    mapping = {g:i for i, g in enumerate(X.columns)}
    #print('mapping: ', mapping)
    TF_sub_numeric = [mapping[t] for t in TF_sub]
    #print('before', X)
    X = np.array(X)
    #print('after', X)
    return X, mapping, theta_true, TF_sub_numeric


def get_filepath():
    savepath = 'simulator/BEELINE-data/'
    filepath = savepath +'beeline_processed.pickle'
    return filepath


def main():
    real_data_names = ['mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L', 'hESC', 'hHep']
    network_names = ['STRING-network', 'Non-Specific-ChIP-seq-network', 'Cell-type-specific-ChIP-seq'] 
    #network_names = ['Cell-type-specific-ChIP-seq'] 
    # in cell type specific, use the cell name to find the corresponding files

    # preparing the real data and save it
    real_data = []
    itr= 0
    for network_name in network_names:
        for data_name in real_data_names:
            print('***********', network_name, data_name)
            # run a method for the particular dataset
            X, mapping, theta_true, TF = get_real_data(network_name, data_name)
            real_data.append([X, [network_name, data_name, mapping], theta_true, TF])
            itr += 1
            
    print('Saving the preprocessed data')
    FILEPATH = get_filepath()
    with open(FILEPATH, 'wb') as handle:
        pickle.dump(real_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(FILEPATH, 'rb') as handle:
        load_real_data = pickle.load(handle)
    #print('after: ', load_all_data[0][0])
    print('Files saved')    

    
if __name__=="__main__":
    main()



