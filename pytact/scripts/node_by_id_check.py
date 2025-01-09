from pytact import data_reader, graph_visualize_browse
import pathlib

from typing import Optional, List, DefaultDict
from pytact.data_reader import Node
from pytact.graph_api_capnp_cython import EdgeClassification
from pytact.graph_api_capnp_cython import Graph_Node_Label_Which
from collections import defaultdict 

def get_node_id(node):
    prefix = f"{node.graph}-{node.nodeid}"
    return prefix

def main(): 
    dataset_location = '../../../chicken-game-main/v15-stdlib-coq8.11/dataset'
    # dataset_location = sys.argv[1]
    dag: DefaultDict[str, int] = defaultdict(int) 
    
    with data_reader.data_reader(pathlib.Path(dataset_location)) as reader:
        datapath = pathlib.Path("coq-tactician-stdlib.8.11.dev/theories/Init/Logic.bin")
        peano_dataset = reader[datapath] 
        rel_label = peano_dataset.node_by_id(188).label.which
        pdl = peano_dataset.lowlevel
        size = len(pdl.graph.nodes)
        print(size)
    
        # Define Graph to Traverse
        index = 33
        current_node = peano_dataset.node_by_id(index)
        
        # Initialize the required data structures
        # Define initial variables
        node_type_list: List[Graph_Node_Label_Which] = []
        edge_tupples_with_type_list: List[tuple[int,int,Optional[EdgeClassification]]] = [] #paretnt_id, child_id, type
        queue: list[tuple[Node, Optional[int], Optional[EdgeClassification]]] = [(current_node, None, None)] #current_node, parent_id, edge_type
    
        
        # Limit the traversal to a maximum depth or iterations
        iteration = 1
        max_iterations = 100
        while len(queue) and  iteration < max_iterations:
            iteration += 1
            
            # Dequeue the next node
            current_node, parent_id, edge_type = queue.pop(0)
        
            # If the node is in the graph then
            # Add Edge, Dont expand Children, Dont add Node 
            #if current_node.get_node_id()
            current_node_pytac_id = get_node_id(current_node)
            if current_node_pytac_id in dag: 
                # Add current existing node details
                edge_tupples_with_type_list.append((parent_id, dag[current_node_pytac_id], edge_type))
                continue
                
            else: 
                current_node_id = len(node_type_list)
                dag[current_node_pytac_id] = current_node_id
                # Add current node details
                edge_tupples_with_type_list.append((parent_id, current_node_id, edge_type))
                node_type_list.append(current_node.label.which)
                print((current_node.label.which,current_node_pytac_id))
                # Process children of the current node
                children = list(current_node.children)
                if current_node.label.which == rel_label: 
                    continue
                for edge_label, child_node in children:
                    # Add child and edge details
                    queue.append((child_node, current_node_id, edge_label))
        
if __name__ == "__main__":
    exit(main())

# def main():
#     #dataset_location = '../../../chicken-game-main/v15-stdlib-coq8.11/dataset'
#     dataset_location = sys.argv[1]
#     datapath = pathlib.Path("coq-tactician-stdlib.8.11.dev/theories/Init/Logic.bin")
#     with data_reader.data_reader(pathlib.Path(dataset_location)) as reader:
#         peano_dataset = reader[datapath] 
#         pdl = peano_dataset.lowlevel
#         size = len(pdl.graph.nodes)
#         print(size)
#         print(peano_dataset.node_by_id(0).label.which)
#         print("Line 1:   ", peano_dataset.node_by_id(0).label)
#         print("Line 4:   ", peano_dataset.node_by_id(size-1).label)
#         print("Line 1:   ", list(peano_dataset.node_by_id(0).children))
#         print("Line 4:   ", list(peano_dataset.node_by_id(size-1).children))

# Walk through all the files and walk through all the nodes. 
# To extract the examples you get all the childern 
# Numbr of nodes and edge types in the dataset 
