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
        # X = normalize(X)
        return X
 

 

def create_batch_from_groups(groups, qry_size=1, pos_size=1, neg_size=1, seen_query_ids=None): 
    queries, candidates, targets = [], [], []
    for group in groups: 
        if seen_query_ids is not None:
            seen_query_ids.update(set(list(group["query"].keys())))
        target = len(queries)
        # queries.append(random.choice(list(group["query"].values())))
        
        targets += [target] * qry_size        
        queries += random.choices(list(group["query"].values()), k=qry_size)
        candidates.append(random.choice(list(group["positive"].values())))  # a single positive 
        candidates += random.choices(list(group["negative"].values()), k=neg_size) 
        
        

    assert len(queries) == len(targets) == len(groups) * qry_size
    assert len(candidates) == len(groups) * (1+neg_size)
    return np.array(queries), np.array(candidates), np.array(targets)


class BiEncoderIndexManager(BartIndexManager):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.logger = logger
        self.name = "biencoder_index_manager"
        self.query_input_dim = 768*2*2
        self.memory_input_dim = 768*2
        self.hidden_dim = 512
        self.final_index_dim = 256
        self.memory_encoder = None
        self.query_encoder = None
    
    def init_biencoder_modules(self):
        self.memory_encoder = MLP(self.memory_input_dim, self.final_index_dim, self.hidden_dim)
        self.query_encoder = MLP(self.query_input_dim, self.final_index_dim, self.hidden_dim*2)
    
    def train_biencoder(self, train_data, eval_data):
        lr = 5e-2 # self.index_train_args.learning_rate
        
        n_steps = 500
        trainable_params = list(self.query_encoder.parameters()) + list(self.memory_encoder.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=lr) 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.1, last_epoch=-1)

        # margin = 0.3
        # loss_fn = torch.nn.TripletMarginLoss(margin=margin)
        # loss_fn = MyTripletLoss(margin=margin)
        
        eval_per_steps = 10
        batch_size = 4
        qry_size = 8
        neg_size = 8 
        seen_query_ids = set()

        best_eval_acc = self.eval_func(eval_data, k=8, seen_query_ids=seen_query_ids)
        self.logger.info(f"Zero-Training Evaluation Top-k acc: {best_eval_acc}")
        

        losses = []
        for _step in tqdm(range(n_steps), desc="Training steps"):
            self.memory_encoder.train()
            self.query_encoder.train()  
            # self.logger.info(f"Training step {_step}/{n_steps}")
             
            sampled_groups = random.choices(train_data, k=batch_size)
            queries, candidates, targets = create_batch_from_groups(sampled_groups, qry_size=qry_size, neg_size=neg_size, seen_query_ids=seen_query_ids)
            optimizer.zero_grad()  
            
            query_inputs = self.query_encoder(torch.Tensor(queries))
            memory_inputs = self.memory_encoder(torch.Tensor(candidates)) 

            scores = torch.matmul(query_inputs, memory_inputs.transpose(0, 1))
            loss = F.cross_entropy(scores, torch.LongTensor(targets), reduction="mean") 
            

            # self.logger.info(f"loss.item()={loss.item()};") 
            losses.append(loss.item())
            loss.backward()

            optimizer.step() 
            scheduler.step() 
            self.logger.info(f"---- Completed epoch with avg training loss {sum(losses)/len(losses)}.")

            # self.query_encoder.eval()
            # self.memory_encoder.eval()
            # updated_query_inputs = self.query_encoder(torch.Tensor(queries))
            # updated_memory_inputs = self.memory_encoder(torch.Tensor(candidates))
            # updated_scores = torch.matmul(updated_query_inputs, updated_memory_inputs.transpose(0, 1))
            # updated_loss = F.cross_entropy(updated_scores, torch.LongTensor(targets), reduction="mean")

            # self.logger.info(f"updated_loss.item()={updated_loss.item()};")


            # self.logger.info(f"self.query_encoder.layers[0].weight = {self.query_encoder.layers[0].weight}")        
            # self.logger.info(f"self.memory_encoder.layers[0].weight = {self.memory_encoder.layers[0].weight}")        
            if _step % eval_per_steps == 0:
                eval_acc = self.eval_func(eval_data, k=8, seen_query_ids=seen_query_ids)
                best_eval_acc = max(best_eval_acc, eval_acc)
                self.logger.info(f"Evaluation Top-8 acc @ {_step}: {eval_acc} | best_eval_acc={best_eval_acc}")



    def eval_func(self, eval_data, k=5, seen_query_ids=None, filter=False):
        top_k_accs = []
        # eval_data = eval_data[:10] # DEBUG: testing
        tested_query_ids = set()
        for item in eval_data:
            query_vectors = []
            query_ids = []
            for qry_id, qry_vec in item["query"].items():
                if filter and seen_query_ids is not None and qry_id  in seen_query_ids:
                    # Remove the seen qry ids
                    continue
                query_ids.append(qry_id)
                query_vectors.append(qry_vec)
            
            if len(query_ids) == 0:
                continue
            
            tested_query_ids.update(query_ids)
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
            
            self.query_encoder.eval()
            self.memory_encoder.eval()
            q = self.query_encoder(torch.Tensor(query_vectors)).detach().numpy()
            m = self.memory_encoder(torch.Tensor(all_candidate_vectors)).detach().numpy()
            memory_index = faiss.IndexFlatL2(m.shape[1])
            memory_index.add(m)
            Ds, Is = memory_index.search(q, k) 
            for index_list in Is:
                retrieved_top_ids = [all_candidaites[ind] for ind in index_list]
                top_k_accs.append(len([x for x in retrieved_top_ids if x in positive_ids])/k)
            
            del memory_index
        if seen_query_ids is not None:
            coverage = len(tested_query_ids & seen_query_ids)/len(tested_query_ids)
            print(f"#tested_query_ids={len(tested_query_ids)}; coverage={coverage}")
        # print(f"top_k_accs = {len(top_k_accs)}; ")
        return np.mean(top_k_accs)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)


    seed = 42
    set_seeds(seed)
    train_data = load_distant_supervision("exp_results/supervision_data/1009_dm_simple/", sample_size=20, logger=logger) #, specified_names=["dm.30-1", "dm.30-4", "dm.29-3", "dm.29-2", "dm.29-1", "dm.29-0"])
       
    logger.info(f"num_groups = {len(train_data)}")

    eval_data = load_distant_supervision("exp_results/supervision_data/1009_dm_simple/", sample_size=5, logger=logger) #, specified_names=["dm.32-0"])

    biencoder_memory_module = BiEncoderIndexManager(logger)
    biencoder_memory_module.init_biencoder_modules()  
    biencoder_memory_module.train_biencoder(train_data, eval_data)
 


