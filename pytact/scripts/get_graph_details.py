# Load the dataset into PyTactician's visualizer.
from pytact import data_reader, graph_visualize_browse
import pathlib
from typing import Optional, List, DefaultDict, List, Tuple, Dict
from pytact.data_reader import Node
from pytact.graph_api_capnp_cython import EdgeClassification
from pytact.graph_api_capnp_cython import Graph_Node_Label_Which
from collections import defaultdict
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
import random
import numpy as np
from sklearn.metrics import classification_report

def get_file_size(reader, dataset_pointer): 
        pdl = dataset_pointer.lowlevel
        size = len(pdl.graph.nodes)
        return size

def get_node_id(node):
    prefix = f"{node.graph}-{node.nodeid}"
    return prefix

def get_pytac_node_id(index, reader, dataset_pointer):
    return get_node_id(dataset_pointer.node_by_id(index))

def get_graph_nodes_depth(index, reader, dataset_pointer):    
    depth = defaultdict(int)  # None indicates not yet processed
    visited = defaultdict(bool)
    stack = []  
    current_node = dataset_pointer.node_by_id(index)
    # Process initial node 

    stack.append((current_node, 0))
    while stack: 
        current_node, state = stack.pop()
        node_pytac_id = get_node_id(current_node)
        if state == 0:  # First time visiting the node
                stack.append((current_node, 1))  # Push back with state 1
                # Add all children to the stack
                for _, child in list(current_node.children):
                    if get_node_id(child) not in visited:  # Process only unprocessed children
                        stack.append((child, 0))
                        visited[get_node_id(child)]=True
        else: # all children processed
            if len(list(current_node.children))==0 or current_node.label.which.name=='REL': 
                depth[node_pytac_id] = 0
            else:
                depth[node_pytac_id] = max([depth[get_node_id(child)] for _,child in list(current_node.children)])+1
    return depth 

def get_graph_details(index, reader, dataset_pointer) -> Tuple[List[str], List[str], List[Tuple[int, int]], List[int], List[int]]:    
    """
    Performs a depth-first search (DFS) on a graph starting from a specified node, collecting details about nodes and edges,
    and determining an execution order for nodes that is not 'REL' type.

    Parameters:
        index (int): The index of the root node to start DFS.
        reader: Data reader object (potentially unused and can be removed if not needed elsewhere).
        dataset_pointer: Interface to access nodes by ID.

    Returns:
        Tuple containing lists of:
        - Node attributes
        - Edge attributes
        - Edge indices mapped through node_to_index
        - Order of execution values
        - List of node indices as encountered
    """
    
    #Initialize datastructures
    node_attr: List[str] = []
    node_index: List[int] = []
    node_to_index: Dict[int] = {}
    edge_attr: List[str] = []
    edge_index: List[tuple(str,str)] = []
    order_of_execution = defaultdict(int)  
    
    # Intialize stack and visited dict for DFS initialization and process initial node
    stack = []  
    visited = defaultdict(bool)
    stack.append((dataset_pointer.node_by_id(index), 0))
    i = 0
    while stack: 
        current_node, state = stack.pop()
        current_node_id = get_node_id(current_node)

        if state == 0:  # First time visiting the node
            stack.append((current_node, 1))  # Push back with state 1
            node_to_index[current_node_id] = len(node_attr)
            node_attr.append(current_node.label.which.name)
            node_index.append(current_node_id)
            # Add all children to the stack
            for edge_label, child in list(current_node.children):
                child_id = get_node_id(child)
                if not visited[child_id]:  # Process only unprocessed children
                    visited[child_id]=True
                    if current_node.label.which.name!='REL':
                        stack.append((child, 0))
                        edge_index.append((current_node_id, child_id))
                        edge_attr.append(edge_label.name)
        else: # all children processed
            if len(list(current_node.children))==0 or current_node.label.which.name=='REL': # Dont process the REL nodes for now.
                order_of_execution[current_node_id] = 0
            else:
                order_of_execution[current_node_id] = max([order_of_execution[get_node_id(child)] for _,child in list(current_node.children)])+1
                
    return node_attr, edge_attr, [(node_to_index[i], node_to_index[j]) for i, j in edge_index], [order_of_execution[i] for i in node_index], node_index


# Define a custom dataset class with an option to save and load
class PytacPygeomDataset(Dataset):
    def __init__(self, root, reader, files_list, transform=None, pre_transform=None):
        super().__init__(root, reader, files_list, transform, pre_transform)
        self.root = root
        self.graphs = []
        self.reader = reader
        self.files_list = files_list
    
    def _process(self):
        # Example: Generate some graphs
        for filename in self.files_list:    
            dataset_pointer = self.reader[pathlib.Path(filename)]
            size = get_file_size(self.reader, dataset_pointer)
 

            #for index in range(size):
            for index in range(10):    
                if not index%100: 
                    print(index)
                dag_info = get_graph_details(index, self.reader, dataset_pointer)
                
                x = dag_info[0] # node_attributes
                edge_index = dag_info[1] 
                edge_attr = dag_info[2]
                execution_order = dag_info[4] # execution order for the TreeLstm from low to high - first Leaf nodes
                print(x, edge_attr, execution_order)
                self.graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, execution_order=execution_order))

            # Save the processed data
        self.save_to_disk()
    
    def save_to_disk(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        for idx, graph in enumerate(self.graphs):
            path = os.path.join(self.processed_dir, f'a_graph_{idx}.pt')
            torch.save(graph, path)
    
    def load_from_disk(self):
        self.graphs = []
        for filename in os.listdir(self.processed_dir):
            if filename.endswith('.pt'):
                path = os.path.join(self.processed_dir, filename)
                graph = torch.load(path)
                self.graphs.append(graph)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]



def main(): 
    print("Initial test: ......")
    dataset_location_ = '../../../../v15-stdlib-coq8.11/dataset'
    file_name="coq-tactician-stdlib.8.11.dev/theories/Init/Logic.bin"
    peano_filename = "coq-tactician-stdlib.8.11.dev/theories/Init/Peano.bin"
    dataset_path = pathlib.Path(dataset_location_)
    file_path = pathlib.Path(file_name)
    with data_reader.data_reader(dataset_path) as reader: 
        dataset_pointer = reader[file_path] 
        print(get_file_size(reader=reader, dataset_pointer=dataset_pointer))
        print(get_pytac_node_id(index=0, reader=reader, dataset_pointer=dataset_pointer))
        print(get_graph_nodes_depth(index=0, reader=reader, dataset_pointer=dataset_pointer))
        print(get_graph_details(index=0, reader=reader, dataset_pointer=dataset_pointer))
        print("Initial test: Done")
        
        print("Dataset Creation: ......")
        root_dir = './graph_dataset'
        dataset = PytacPygeomDataset(root_dir, reader=reader, files_list=[peano_filename])
        
        dataset._process()  # Create and save the dataset
        print("Dataset Creation: Done")
        
if __name__ == "__main__":
    exit(main())


