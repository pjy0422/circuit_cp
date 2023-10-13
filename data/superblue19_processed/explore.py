import dgl
import numpy as np
import torch
from ogb.io import DatasetSaver
import pickle as pkl
import inspect


data = pkl.load(open('hetero_900.pickle', 'rb'))
print(data[0][0].ndata['label'])
dataset_name = 'ogbn-superblue19'
saver = DatasetSaver(dataset_name=dataset_name, is_hetero=False, version=1)
graph_list = []
graph = dict()
graph['edge_index'] = np.array(data[0][0].edges())
graph['node_feat'] = np.array(data[0][0].ndata['feat'])
graph['node_pos'] = np.array(data[0][0].ndata['pos'])
graph['num_nodes'] = data[0][0].num_nodes()
graph_list.append(graph)
saver.save_graph_list(graph_list)
labels = np.array(data[0][0].ndata['label']).reshape(-1, 1)
saver.save_target_labels(labels)
