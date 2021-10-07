from semanticdebugger.debug_algs.index_based.index_manager import BartIndexManager

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
import random 
import glob
import os
import pickle
from tqdm import tqdm
import numpy as np

def normalize(inputs):
    return F.normalize(inputs, dim=0)

def build_triplets(query_vectors, positive_vectors, negative_vectors, size=8):
    n_queries = query_vectors.shape[0]
    n_positive = positive_vectors.shape[0]
    n_negative = negative_vectors.shape[0]
 
    query_inputs = query_vectors.repeat(size, axis=0) 
    positive_indices = list(range(n_positive))
    negative_indices = list(range(n_negative))

    selected_positive_indices = random.choices(positive_indices, k=size*n_queries)
    selected_negative_indices = random.choices(negative_indices, k=size*n_queries)

    positive_inputs = positive_vectors[selected_positive_indices]
    negative_inputs = negative_vectors[selected_negative_indices]
 
    # assert query_inputs == positive_inputs) == negative_inputs) == size * n_queries
    # print(f"query_inputs.shape={query_inputs.shape}")
    # print(f"positive_inputs.shape={positive_inputs.shape}")
    # print(f"negative_inputs.shape={negative_inputs.shape}")
    return query_inputs, positive_inputs, negative_inputs


def load_distant_supervision(folder_name): 
    pkl_files = glob.glob(os.path.join(folder_name, '*.pkl'))[:]
    ds_items = []
    for pkl_path in tqdm(pkl_files, desc="loading pkl files"):
        with open(pkl_path, "rb") as f:
            ds_items += pickle.load(f)
    return ds_items 
    


class MLP(Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
          torch.nn.Flatten(),
          torch.nn.Linear(input_dim, hidden_dim),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_dim, output_dim),
          torch.nn.ReLU()
        )

    def forward(self, X):
        X = self.layers(X)
        return X

class BiEncoderIndexManager(BartIndexManager):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.logger = logger
        self.name = "biencoder_index_manager"
        self.query_input_dim = 768*2*2
        self.memory_input_dim = 768*2
        self.hidden_dim = 1024
        self.final_index_dim = 256
        self.memory_encoder = None
        self.query_encoder = None
    
    def init_biencoder_modules(self):
        self.memory_encoder = MLP(self.query_vector_dim, self.final_index_dim, self.hidden_dim)
        self.query_encoder = MLP(self.memory_input_dim, self.final_index_dim, self.hidden_dim)
    
    def train_biencoder(self, train_data):
        lr = 1e-3 # self.index_train_args.learning_rate
        margin = 1.0
        n_epochs = 20
        memory_optimizer = torch.optim.Adam(self.memory_encoder.parameters(), lr=lr)
        query_optimizer = torch.optim.Adam(self.query_encoder.parameters(), lr=lr)
        memory_scheduler = torch.optim.lr_scheduler.StepLR(memory_optimizer, 8, gamma=0.1, last_epoch=-1)
        query_scheduler = torch.optim.lr_scheduler.StepLR(query_optimizer, 8, gamma=0.1, last_epoch=-1)
        loss_fn = torch.nn.TripletMarginLoss(margin=margin)

        self.memory_encoder.train()
        self.query_encoder.train() 

        for epoch in range(n_epochs):
            self.logger.info(f"Training epoch {epoch}/{n_epochs}")
            losses = []
            random.shuffle(train_data)
            # Iterate over all sets of query/example triplets
            for batch_idx, (query_examples, positive_examples, negative_examples) in enumerate(train_data):
                memory_optimizer.zero_grad()
                query_optimizer.zero_grad()

                query_inputs, positive_inputs, negative_inputs = \
                    build_triplets(query_examples, positive_examples, negative_examples)

                loss = loss_fn(query_inputs, positive_inputs, negative_inputs)

                losses.append(loss.item())
                loss.backward()

                memory_optimizer.step()
                query_optimizer.step()

            self.logger.info(f"Completed epoch with avg training loss {sum(losses)/len(losses)}.")
            memory_scheduler.step()
            query_scheduler.step()
    


if __name__ == '__main__':
    ds_items = load_distant_supervision("exp_results/supervision_data/1006v3_dm_simple/")
    base_path = "exp_results/supervision_data/1006v3_dm_simple/"
    triplet_per_example = 8
    all_query_vectors = []
    all_pos_vectors = []
    all_neg_vectors = []
    for item in ds_items:
        query_vectors = [v for k,v in item["query"].items()]
        positive_vectors = [v for k,v in item["positive"].items()]
        negative_vectors = [v for k,v in item["negative"].items()]
        query_vectors = np.array(query_vectors)
        positive_vectors = np.array(positive_vectors)
        negative_vectors = np.array(negative_vectors)
        q, p, n = build_triplets(query_vectors, positive_vectors, negative_vectors, triplet_per_example)
        all_query_vectors += list(q)
        all_pos_vectors += list(p)
        all_neg_vectors += list(n)
    
    print(len(ds_items)) 
    all_query_vectors = np.array(all_query_vectors)
    all_pos_vectors = np.array(all_pos_vectors)
    all_neg_vectors = np.array(all_neg_vectors)
    print(f"all_query_vectors.shape: {all_query_vectors.shape}")
    print(f"all_pos_vectors.shape: {all_pos_vectors.shape}")
    print(f"all_neg_vectors.shape: {all_neg_vectors.shape}")

    np.save(os.path.join(base_path, "all_query_vectors.npy"), all_query_vectors)
    np.save(os.path.join(base_path, "all_pos_vectors.npy"), all_pos_vectors)
    np.save(os.path.join(base_path, "all_neg_vectors.npy"), all_neg_vectors)
    
