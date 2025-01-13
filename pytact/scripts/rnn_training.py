# Load the dataset into PyTactician's visualizer.
from pytact import data_reader, graph_visualize_browse
import pathlib
from typing import Optional, List, DefaultDict
from pytact.data_reader import Node
from pytact.graph_api_capnp_cython import EdgeClassification
from pytact.graph_api_capnp_cython import Graph_Node_Label_Which
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import classification_report


class BasicRNN(nn.Module):
    def __init__(self, n_inputs, n_neurons):
        super(BasicRNN, self).__init__()

        self.Wx = torch.randn(n_inputs, n_neurons) # n_inputs X n_neurons
        self.Wh = torch.randn(n_neurons, n_neurons) # n_neurons X n_neurons

        self.b = torch.zeros(1, n_neurons) # 1 X n_neurons

    def forward(self, x, hidden):
        return torch.tanh(torch.mm(x, self.Wx) + torch.mm(hidden, self.Wh) + self.b)
    
    
class BasicCSRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, vocab_size, edges_size):
        super(BasicCSRNN, self).__init__()
        self.Wx = torch.randn(input_size, hidden_size) # n_inputs X n_neurons
        self.We = torch.randn(edges_size, 1, hidden_size) # n_edges X 1 X n_neurons
        self.Wh = torch.randn(hidden_size, hidden_size) # n_neurons X n_neurons
        self.hidden_size =hidden_size
        self.b = torch.zeros(1, hidden_size) # 1 X n_neurons

    def forward(self, embedding, node):
        return self.node_forward(embedding, node)

    def node_forward(self, embedding, node):
        x = embedding.unsqueeze(0)
        if node.children and not node.label.which.name == 'REL':
            hidden = torch.sum(torch.stack([self.node_forward(embedding, child)*self.We[edge_type.value] for edge_type, child in list(node.children)]), dim=0)
            #hidden = torch.sum(torch.stack(self.node_forward(child) for child in node.children), dim=0)
        else:
            # Ensure that the zero tensor is of the correct shape [batch size, hidden size]
            hidden = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float, device=x.device)
        return torch.tanh(torch.mm(x, self.Wx) + hidden + self.b)


class BasicCSRNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, vocab_size, edges_size, output_size):
        super(BasicCSRNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cs_rnn = BasicCSRNN(input_size, hidden_size, embedding_dim, vocab_size, edges_size)
        self.fc = nn.Linear(hidden_size, output_size)  # Linear classification layer

    def forward(self, node):
        # x shape: (batch_size, sequence_length, input_size)
        emb = self.embedding(torch.tensor(node.label.which.value))
        hn = self.cs_rnn(emb, node)  # hn is the last hidden state
        # output shape: (batch_size, sequence_length, hidden_size)
        # hn shape: (1, batch_size, hidden_size)
        hn = hn.squeeze(0)  # Remove the first dimension to match input of linear layer
        logits = self.fc(hn)
        probabilities = F.softmax(logits)  # Applying softmax on the logits for probabilities
        return probabilities

def train_model(model, dataset, epochs, lr, bs, size=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(size):
            graph = dataset.node_by_id(i)
            label = torch.tensor(graph.label.which.value)
            
            optimizer.zero_grad()
            outputs = model(graph)
            
            loss = criterion(outputs, label) / bs
            loss.backward()
            
            if (i + 1) % bs == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / size}')
    return model

def evaluate_model(model, dataset, size):
    predictions, actuals = [], []
    for i in range(size):
        graph = dataset.node_by_id(i)
        label = torch.tensor(graph.label.which.value)
        output = model(graph)
        predicted_label = torch.argmax(output, dim=0)
        predictions.append(predicted_label.item())
        actuals.append(label)
    
    return classification_report(actuals, predictions, output_dict=True)

def main():       
    # Constants and configurations
    DATASET_PATH = '../../../../v15-stdlib-coq8.11/dataset'
    FILE_PATH = "coq-tactician-stdlib.8.11.dev/theories/Init/Logic.bin"
    DATASET_PATH = pathlib.Path(DATASET_PATH)
    FILE_PATH = pathlib.Path(FILE_PATH)
    BATCH_SIZE = 10
    EPOCHS = 5
    LEARNING_RATE = 0.001
    RANDOM_SEED = 42

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    model = BasicCSRNNClassifier(10,20,10,100,100,100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with data_reader.data_reader(DATASET_PATH) as reader: 
        dataset_pointer = reader[FILE_PATH] 
        num_samples = 20
        model = train_model(model, dataset_pointer, EPOCHS, LEARNING_RATE, BATCH_SIZE, num_samples)
        res = evaluate_model(model, dataset_pointer, num_samples)   
    print(res)
    
if __name__ == "__main__":
    exit(main())