from logging import Logger
import logging

from tqdm.utils import disp_trim
from semanticdebugger.debug_algs.index_based.index_manager import BartIndexManager

import torch
from torch import Tensor, normal
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
import random 
import glob
import os
import pickle
from tqdm import tqdm
import numpy as np
import faiss

from semanticdebugger.models.utils import set_seeds

def normalize(inputs):
    return F.normalize(inputs, dim=0)

class MyTripletLoss(torch.nn.Module):
    """
    Triplet loss function.
    """

    def __init__(self, margin=1.0):
        super(MyTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):

        squarred_distance_1 = (anchor - positive).pow(2).sum(1)
        
        squarred_distance_2 = (anchor - negative).pow(2).sum(1)
        
        triplet_loss = F.relu( self.margin + squarred_distance_1 -squarred_distance_2 ).mean()
        
        return triplet_loss, squarred_distance_1, squarred_distance_2

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
    # - 1 # DEBUG: 
    
    # assert query_inputs == positive_inputs) == negative_inputs) == size * n_queries
    # print(f"query_inputs.shape={query_inputs.shape}")
    # print(f"positive_inputs.shape={positive_inputs.shape}")
    # print(f"negative_inputs.shape={negative_inputs.shape}")
    return query_inputs, positive_inputs, negative_inputs


def load_distant_supervision(folder_name, sample_size=1, logger=None, specified_names=None): 
    pkl_files = glob.glob(os.path.join(folder_name, '*.pkl'))[:]
    if specified_names:
        pkl_files = [p for p in pkl_files if p.split("/")[-1].replace(".pkl", "") in specified_names]
    else:
        pkl_files = random.choices(pkl_files, k=sample_size)
    ds_items = []
    logger.info(f"Loading {pkl_files}")
    for pkl_path in tqdm(pkl_files, desc="loading pkl files"):
        with open(pkl_path, "rb") as f:
            ds_items += pickle.load(f)

    # # # For Debugging #
    # for item in ds_items:
    #     for q in item["query"]:
    #         item["query"][q] = np.random.normal(0, 0.3, 768*2*2)
    #     for q in item["positive"]:
    #         item["positive"][q] = np.random.normal(0, 0.1, 768*2)
    #     for q in item["negative"]:
    #         item["negative"][q] = np.random.normal(0, 0.3, 768*2)
    #     pass
    
    return ds_items 
    


class MLP(Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
        #   torch.nn.Flatten(),
          torch.nn.Linear(input_dim, hidden_dim),
          torch.nn.Sigmoid(),
          torch.nn.Linear(hidden_dim, output_dim),

        #   torch.nn.Linear(input_dim, output_dim)
        )
        self.init_weights()

    def init_weights(self):
        for module in self.layers:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_() 
        
    def forward(self, X):
        X = self.layers(X) 
        return X

# class BiMLPEncoder(Module):
#     def __init__(self, query_input_dim, memory_input_dim, final_index_dim, hidden_dim):
#         super().__init__()

#         self.query_input_dim = query_input_dim
#         self.memory_input_dim = memory_input_dim
#         self.final_index_dim = final_index_dim
#         self.hidden_dim = hidden_dim
#         self.memory_encoder = MLP(self.memory_input_dim, self.final_index_dim, self.hidden_dim)
#         self.query_encoder = MLP(self.query_input_dim, self.final_index_dim, self.hidden_dim)
    
#     def forward(self, x, y):
#         q = self.query_encoder(x)
#         m = self.memory_encoder(y)
#         return q, m



def merge_groups(groups):
    q, p, n = None, None, None
    for group in groups:
        if q is None:
            q, p, n = group["q"], group["p"], group["n"]
        else:
            q = np.append(q, group["q"], axis=0)
            p = np.append(p, group["p"], axis=0)
            n = np.append(n, group["n"], axis=0)
    return q, p, n

def create_batch_from_groups(groups, neg_size=1): 
    queries, candidates, targets = [], [], []
    for group in groups: 
        # Correct #
        queries.append(random.choice(group["q"]))
        candidates.append(random.choice(group["p"]))
        candidates += random.choices(group["n"], k=neg_size) 
        
        targets.append(len(queries)-1)

    assert len(queries) == len(groups)
    assert len(candidates) == len(groups) * (1+neg_size)
    return np.array(queries), np.array(candidates), np.array(targets)


class BiEncoderIndexManager(BartIndexManager):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.logger = logger
        self.name = "biencoder_index_manager"
        self.query_input_dim = 768*2*2
        self.memory_input_dim = 768*2
        self.hidden_dim = 64
        self.final_index_dim = 32
        self.memory_encoder = None
        self.query_encoder = None
    
    def init_biencoder_modules(self):
        self.memory_encoder = MLP(self.memory_input_dim, self.final_index_dim, self.hidden_dim)
        self.query_encoder = MLP(self.query_input_dim, self.final_index_dim, self.hidden_dim)
    
    def train_biencoder(self, train_data, eval_data):
        lr = 1e-2 # self.index_train_args.learning_rate
        
        n_steps = 100
        trainable_params = list(self.query_encoder.parameters()) + list(self.memory_encoder.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=lr) 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.1, last_epoch=-1)

        # margin = 0.3
        # loss_fn = torch.nn.TripletMarginLoss(margin=margin)
        # loss_fn = MyTripletLoss(margin=margin)
        

        batch_size = 4
        neg_size = 8

        eval_acc = eval_func(biencoder_memory_module, eval_data, k=5)
        self.logger.info(f"Zero-Training Evaluation Top-5 acc: {eval_acc}")

        losses = []
        for _step in tqdm(range(n_steps), desc="Training steps"):
            self.memory_encoder.train()
            self.query_encoder.train()  
            self.logger.info(f"Training step {_step}/{n_steps}")
             
            sampled_groups = random.choices(train_data, k=batch_size)
            queries, candidates, targets = create_batch_from_groups(sampled_groups, neg_size) 
            optimizer.zero_grad()  
            
            query_inputs = self.query_encoder(normalize(torch.Tensor(queries)))
            memory_inputs = self.memory_encoder(normalize(torch.Tensor(candidates)))  

            # loss, dp, dn = loss_fn(query_inputs, )

            scores = torch.matmul(query_inputs, memory_inputs.transpose(0, 1))
            loss = F.cross_entropy(scores, torch.LongTensor(targets), reduction="mean") 
            

            self.logger.info(f"loss.item()={loss.item()};") 
            losses.append(loss.item())
            loss.backward()

            optimizer.step() 
            scheduler.step() 
            self.logger.info(f"---- Completed epoch with avg training loss {sum(losses)/len(losses)}.")

            self.query_encoder.eval()
            self.memory_encoder.eval()
            updated_query_inputs = self.query_encoder(normalize(torch.Tensor(queries)))
            updated_memory_inputs = self.memory_encoder(normalize(torch.Tensor(candidates)))  
            updated_scores = torch.matmul(updated_query_inputs, updated_memory_inputs.transpose(0, 1))
            updated_loss = F.cross_entropy(updated_scores, torch.LongTensor(targets), reduction="mean")

            self.logger.info(f"updated_loss.item()={updated_loss.item()};")


            # self.logger.info(f"self.query_encoder.layers[0].weight = {self.query_encoder.layers[0].weight}")        
            # self.logger.info(f"self.memory_encoder.layers[0].weight = {self.memory_encoder.layers[0].weight}")        

            eval_acc = eval_func(biencoder_memory_module, eval_data, k=5)
            self.logger.info(f"Evaluation Top-5 acc: {eval_acc}")


            
    

def eval_func(biencoder, eval_data, k=5):
    top_k_accs = []
    # eval_data = eval_data[:10] # DEBUG: testing
    for item in eval_data:
        query_vectors = [v for k,v in item["query"].items()]
        positive_ids = set()
        all_candidaites = []
        all_candidate_vectors = []
        for ex_id, vector in item["positive"].items():
            positive_ids.add(ex_id)
            all_candidaites.append(ex_id)
            all_candidate_vectors.append(vector)
        for ex_id, vector in item["negative"].items(): 
            all_candidaites.append(ex_id)
            all_candidate_vectors.append(vector)
            # all_candidate_vectors.append([v-1 for v in vector]) # DEBUG: 
         
        query_vectors = np.array(query_vectors)
        all_candidate_vectors = np.array(all_candidate_vectors)  
        
        biencoder.query_encoder.eval()
        biencoder.memory_encoder.eval()
        q = biencoder.query_encoder(torch.Tensor(query_vectors)).detach().numpy()
        m = biencoder.memory_encoder(torch.Tensor(all_candidate_vectors)).detach().numpy()
        memory_index = faiss.IndexFlatL2(m.shape[1])
        memory_index.add(m)
        Ds, Is = memory_index.search(q, k)
        # TODO: why all I in Is are exactly the same?
        for index_list in Is:
            retrieved_top_ids = [all_candidaites[ind] for ind in index_list]
            top_k_accs.append(len([x for x in retrieved_top_ids if x in positive_ids])/k)
         
        del memory_index
    print(f"top_k_accs = {len(top_k_accs)}")
    return np.mean(top_k_accs)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)


    seed = 42
    set_seeds(seed)
    ds_items = load_distant_supervision("exp_results/supervision_data/1006v3_dm_simple/", sample_size=3, logger=logger, 
                                        specified_names=["dm.29-5"])
                                        # , "dm.29-4", "dm.29-3", "dm.29-2", "dm.29-1", "dm.29-0"])
     
    all_groups = []
    for item in ds_items:
        q = item["query"].values()
        p = item["positive"].values()
        n = item["negative"].values()
        # all_groups.append(dict(q=list(q)[:1], p=list(p)[:1], n=list(n)[:1]))
        all_groups.append(dict(q=list(q), p=list(p), n=list(n)))
    train_data = all_groups 
    logger.info(f"num_groups = {len(all_groups)}")

    eval_data = load_distant_supervision("exp_results/supervision_data/1006v3_dm_simple/", sample_size=1, logger=logger, specified_names=["dm.29-5"])

    biencoder_memory_module = BiEncoderIndexManager(logger)
    biencoder_memory_module.init_biencoder_modules()  
    biencoder_memory_module.train_biencoder(train_data, eval_data)
 


