# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
from logging import Logger
import logging
from torch.cuda import memory

from tqdm.utils import disp_trim
from cmr.debug_algs.index_based.index_manager import BartIndexManager

import torch
from torch import Tensor, combinations, normal
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
import json
import wandb
from cmr.debug_algs.index_based.index_utils import get_bart_dual_representation
from cmr.models.run_bart import train

from cmr.models.utils import set_seeds
from faiss import normalize_L2


def load_distant_supervision(folder_name, sample_size=1, logger=None, specified_names=None, exclude_files=[], train_args=None):
    pkl_files = glob.glob(os.path.join(folder_name, '*.pkl'))[:]
    pkl_files = [f for f in pkl_files if f not in exclude_files] 
    if specified_names:
        pkl_files = [p for p in pkl_files if p.split(
            "/")[-1].replace(".pkl", "") in specified_names]
    else:
        pkl_files = random.choices(pkl_files, k=sample_size)
    
    ds_items = []
    logger.info(f"Loading {pkl_files}")
    for pkl_path in tqdm(pkl_files, desc="loading pkl files"):
        with open(pkl_path, "rb") as f:
            ds_items += pickle.load(f)
    

    for item in ds_items:
        for q in item["query"]:
            original_dim = len(item["query"][q])
            if train_args.query_only_after:
                item["query"][q] = item["query"][q][original_dim//2:]
            if train_args.query_only_before:
                item["query"][q] = item["query"][q][:original_dim//2]
            if train_args.query_delta:
                before = item["query"][q][original_dim//2:]
                after = item["query"][q][:original_dim//2]
                item["query"][q] = before + [i-j for i, j in zip(before, after)]

    # np.random.seed(42)
    # # # For Debugging the data-distribution #
    # print("generating random data")
    # for item in ds_items:
    #     for q in item["query"]:
    #         item["query"][q] = np.random.normal(0, 0.1, 768*2*2) 
    #     for q in item["positive"]:
    #         item["positive"][q] = np.random.normal(0, 0.1, 768*2) 
    #     for q in item["negative"]:
    #         item["negative"][q] = np.random.normal(0.6, 0.1, 768*2) 
    #     pass

    # if exclude_files:
    #     np.random.seed(45)
    #     print("generating purturbs on the test data")
    #     for item in ds_items:
    #         for q in item["query"]:  
    #             item["query"][q] += np.random.normal(0, 5e-2, 768*2*2)
    #         for q in item["positive"]:  
    #             item["positive"][q] += np.random.normal(0, 5e-2, 768*2)
    #         for q in item["negative"]:  
    #             item["negative"][q] += np.random.normal(0, 5e-2, 768*2)
    return ds_items, pkl_files


class MLP(Module):
    def __init__(self, input_dim, output_dim, hidden_dim, droprate=0):
        super().__init__()
        if hidden_dim > 0:
            self.layers = torch.nn.Sequential(
                # torch.nn.Flatten(),
                # nn.BatchNorm1d(input_dim),
                # nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                # nn.LayerNorm(hidden_dim),
                # nn.Sigmoid(),
                nn.Dropout(droprate),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim), 
            )
        else:
            self.layers = torch.nn.Sequential(
                # torch.nn.Flatten(),
                # nn.BatchNorm1d(input_dim),
                # nn.Linear(input_dim, hidden_dim),
                # nn.BatchNorm1d(hidden_dim),
                # nn.ReLU(),
                # nn.Sigmoid(),
                # nn.Dropout(droprate),
                # nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(input_dim),
                # nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim), 
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


def create_batch_from_groups(groups, qry_size=1, pos_size=1, neg_size=1, seen_query_ids=None, seen_memory_ids=None, query_mean=True):
    if query_mean:
        qry_sample_size = qry_size
        qry_size = 1 # effective qry size
    else:
        qry_sample_size = qry_size

    queries, candidates, targets = [], [], []
    for group in groups:
        # TODO: this is overly recorded..
        if seen_query_ids is not None:
            seen_query_ids.update(set(list(group["query"].keys())))
        if seen_memory_ids is not None:
            seen_memory_ids.update((set(list(group["positive"].keys()))))
            seen_memory_ids.update((set(list(group["negative"].keys()))))
        
        # queries.append(random.choice(list(group["query"].values())))
        
        selected_queries = random.choices(list(group["query"].values()), k=qry_sample_size)
        if query_mean:
            selected_queries = np.array(selected_queries)
            queries.append(np.mean(selected_queries, axis=0))
        else:
            queries += selected_queries
        target = len(candidates)
        candidates += random.choices(list(group["positive"].values()), k=pos_size)  # for training, it must be a single positive
        candidates += random.choices(list(group["negative"].values()), k=neg_size)
        if pos_size > 1:
            targets += [list(range(target, target+pos_size))] * qry_size    # N*C
        elif pos_size == 1:
            targets += [target] * qry_size # N*1

    assert len(queries) == len(targets) == len(groups) * qry_size
    assert len(candidates) == len(groups) * (pos_size + neg_size)

    if pos_size > 1:
        return np.array(queries), np.array(candidates), targets
    else:
        return np.array(queries), np.array(candidates), np.array(targets)


class BiEncoderIndexManager(BartIndexManager):
    def __init__(self, logger):
        super().__init__(logger=logger)
        self.logger = logger
        self.name = "biencoder_index_manager"
        self.query_input_dim = 768*2*2
        self.memory_input_dim = 768*2
        self.hidden_dim = 512
        self.dim_vector = 256  # final dim
        self.memory_encoder = None
        self.query_encoder = None
        self.train_args = None

        # cl
        self.before_model = None 
        self.after_model = None 

    def load_encoder_model(self, base_model_args, memory_encoder_path, query_encoder_path):
        super().load_encoder_model(base_model_args)
        if self.memory_encoder is None:
            self.init_biencoder_modules()
            self.memory_encoder.load_state_dict(torch.load(memory_encoder_path))
            self.query_encoder.load_state_dict(torch.load(query_encoder_path))
            self.logger.info(f"Loading bi-encoders.memory_encoder from {memory_encoder_path}")
            self.logger.info(f"Loading bi-encoders.query_encoder from {query_encoder_path}")

    def init_biencoder_modules(self):
        self.query_input_dim = self.train_args.query_input_dim
        self.memory_input_dim = self.train_args.memory_input_dim
        self.hidden_dim = self.train_args.hidden_dim
        self.dim_vector = self.train_args.dim_vector
        self.memory_encoder = MLP(self.memory_input_dim, self.dim_vector,
                                  self.hidden_dim, droprate=self.train_args.droprate)
        self.query_encoder = MLP(self.query_input_dim, self.dim_vector,
                                 self.hidden_dim, droprate=self.train_args.droprate)

    def get_representation(self, examples):
        """only for the memory encoding here"""
        bart_reps = super().get_representation(examples)
        bart_reps = np.array(bart_reps)
        self.memory_encoder.eval()
        all_vectors = self.memory_encoder(torch.Tensor(bart_reps)).detach().numpy()
        return all_vectors

    def train_biencoder(self, train_data, eval_data):
        trainable_params = list(self.query_encoder.parameters()) + \
            list(self.memory_encoder.parameters())
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"# params of query_encoder = {count_parameters(self.query_encoder)}")
        self.logger.info(f"# params of memory_encoder = {count_parameters(self.memory_encoder)}")

        optimizer = torch.optim.Adam(trainable_params, lr=self.train_args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.99, last_epoch=-1)
        gradient_acc_steps = 1

        if self.train_args.use_cuda:
            self.query_encoder.to(torch.device("cuda"))
            self.memory_encoder.to(torch.device("cuda"))

        seen_query_ids = set()
        seen_memory_ids = set()
        eval_at_K = 16
        best_eval_acc = self.eval_func_v1(
            eval_data, k=eval_at_K, seen_query_ids=seen_query_ids, seen_memory_ids=seen_memory_ids)
        self.logger.info(f"Valid Acc @ 0: Top-{eval_at_K} acc: {best_eval_acc}")

        if self.train_args.wandb: 
            wandb.log({"valid_accuracy": best_eval_acc}, step=0)
            wandb.log({"best_valid_acc": best_eval_acc}, step=0)

        losses = []
        no_up = 0
        for _step in tqdm(range(self.train_args.n_steps), desc="Training steps"):
            self.memory_encoder.train()
            self.query_encoder.train()
            # self.logger.info(f"Training step {_step}/{self.train_args.n_steps}")

            sampled_groups = random.choices(train_data, k=self.train_args.batch_size)
            queries, candidates, targets = create_batch_from_groups(
                sampled_groups, 
                qry_size=self.train_args.qry_size,
                pos_size=self.train_args.pos_size, 
                neg_size=self.train_args.neg_size, 
                seen_query_ids=seen_query_ids, seen_memory_ids=seen_memory_ids,
                query_mean=self.train_args.use_query_mean)
            optimizer.zero_grad()


            qry_tensors = torch.Tensor(queries)
            mem_tensors = torch.Tensor(candidates)
            if self.train_args.use_cuda:
                qry_tensors = qry_tensors.to(torch.device("cuda"))
                mem_tensors = mem_tensors.to(torch.device("cuda"))
            query_inputs = self.query_encoder(qry_tensors)
            memory_inputs = self.memory_encoder(mem_tensors)

            scores = torch.matmul(query_inputs, memory_inputs.transpose(0, 1))
            if self.train_args.pos_size == 1:
                tgt_tensors = torch.LongTensor(targets)
                if self.train_args.use_cuda:
                    tgt_tensors = tgt_tensors.to(torch.device("cuda"))
                loss = F.cross_entropy(scores, tgt_tensors, reduction="mean")
            elif self.train_args.pos_size > 1:
                multi_hot_targets = []
                for target in targets:
                    labels = torch.LongTensor(target)
                    labels = labels.unsqueeze(0)
                    multi_hot_targets.append(torch.zeros(labels.size(0), len(candidates)).scatter_(1, labels, 1.))
                multi_hot_targets = torch.stack(multi_hot_targets, dim=1)
                multi_hot_targets = multi_hot_targets.view(scores.size())
                tgt_tensors = torch.Tensor(multi_hot_targets)
                criterion = torch.nn.BCEWithLogitsLoss(reduction="mean") 
                if self.train_args.use_cuda:
                    tgt_tensors = tgt_tensors.to(torch.device("cuda"))
                loss = criterion(scores, tgt_tensors)

            # self.logger.info(f"loss.item()={loss.item()};")
            losses.append(loss.item())
            loss.backward()
            if self.train_args.wandb:
                wandb.log({"lr": float(optimizer.param_groups[0]['lr'])}, step=_step)
                wandb.log({"loss": float(loss)}, step=_step)
                wandb.log({"avg_loss": float(sum(losses)/len(losses))}, step=_step)    

            # clip 
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            optimizer.step()
            scheduler.step()
            # self.logger.info(f"self.query_encoder.layers[0].weight = {self.query_encoder.layers[0].weight}")
            # self.logger.info(f"self.memory_encoder.layers[0].weight = {self.memory_encoder.layers[0].weight}")
            if _step > 0 and _step % self.train_args.eval_per_steps == 0:
                self.logger.info(f"---- Completed epoch with avg training loss {sum(losses)/len(losses)}.")
                train_acc = self.eval_func_v1(train_data[:], k=eval_at_K)

                self.logger.info(
                    f"Train Acc: Top-{eval_at_K} acc @ {_step}: {train_acc} | ")

                valid_acc = self.eval_func_v1(eval_data, k=eval_at_K, seen_query_ids=seen_query_ids, seen_memory_ids=seen_memory_ids)

                best_eval_acc = max(best_eval_acc, valid_acc)
                
                if self.train_args.wandb:
                    wandb.log({"train_accuracy": train_acc}, step=_step)
                    wandb.log({"valid_accuracy": valid_acc}, step=_step)
                    wandb.log({"best_valid_acc": best_eval_acc}, step=_step)
                self.logger.info(
                    f"Valid ACc: Top-{eval_at_K} acc @ {_step}: {valid_acc} | best_eval_acc={best_eval_acc}")
                


                if best_eval_acc == valid_acc:
                    self.logger.info("new record; saving the biencoder ckpts.")
                    no_up = 0
                elif best_eval_acc > valid_acc:
                    no_up += 1
                    if no_up >= self.train_args.patience:
                        break
                    if self.train_args.save_ckpt:
                        self.save_biencoder()



    def eval_func_v2(self, eval_data, k=None, seen_query_ids=None, seen_memory_ids=None, filter=False):
        # based on pair-wise comparisions 
        self.query_encoder.eval()
        self.memory_encoder.eval()
        eval_scores = []
        for group in eval_data:
            queries, candidates, targets = create_batch_from_groups([group], qry_size=16, pos_size=8, neg_size=8)
            
            
            # query_inputs = self.query_encoder(torch.Tensor(queries))
            # memory_inputs = self.memory_encoder(torch.Tensor(candidates))

            qry_tensors = torch.Tensor(queries)
            mem_tensors = torch.Tensor(candidates)
            if self.train_args.use_cuda:
                qry_tensors = qry_tensors.to(torch.device("cuda"))
                mem_tensors = mem_tensors.to(torch.device("cuda"))
            query_inputs = self.query_encoder(qry_tensors)
            memory_inputs = self.memory_encoder(mem_tensors)

            scores = torch.matmul(query_inputs, memory_inputs.transpose(0, 1))
            querywise_scores = []
            for qid in range(len(queries)):
                pairwise_comp = []
                pos_start = 0   # always 0
                pos_end = pos_start + 8
                neg_start = pos_end
                neg_end = neg_start + 8
                for pos_ind in range(pos_start, pos_end):
                    for neg_ind in range(neg_start, neg_end):
                        score_pos = scores[qid][pos_ind]
                        score_neg = scores[qid][neg_ind]
                        pairwise_comp.append(int(score_pos > score_neg))
                pairwise_score = np.mean(pairwise_comp)
                querywise_scores.append(pairwise_score)
            group_score = np.mean(querywise_scores)
            eval_scores.append(group_score)
        return np.mean(eval_scores)
 
    def eval_func_v1(self, eval_data, k=5, seen_query_ids=None, seen_memory_ids=None, filter=False):
        top_k_accs = []
        tested_query_ids = set()
        tested_memory_ids = set()
        for item in eval_data:
            query_vectors = []
            query_ids = []
            for qry_id, qry_vec in item["query"].items():
                if filter and seen_query_ids is not None and qry_id in seen_query_ids:
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
            
            memory_items = list(item["negative"].items()) + list(item["positive"].items())
            random.shuffle(memory_items)    # to avoid the case where they have the same scores
            for ex_id, vector in memory_items:
                all_candidaites.append(ex_id)
                tested_memory_ids.add(ex_id)
                all_candidate_vectors.append(vector)
                if filter and seen_memory_ids is not None and ex_id in seen_memory_ids:
                    # Remove the seen memory ids
                    continue
                all_candidaites.append(ex_id)
                tested_memory_ids.add(ex_id)
                all_candidate_vectors.append(vector)
                # all_candidate_vectors.append([v-1 for v in vector]) # DEBUG:

            query_vectors = np.array(query_vectors)
            all_candidate_vectors = np.array(all_candidate_vectors)

            self.query_encoder.eval()
            self.memory_encoder.eval()
            q_inputs = torch.Tensor(query_vectors)
            m_inputs = torch.Tensor(all_candidate_vectors)
            if self.train_args.use_cuda:
                q_inputs = q_inputs.to(torch.device("cuda"))
                m_inputs = m_inputs.to(torch.device("cuda"))

            q = self.query_encoder(q_inputs).detach().cpu().numpy()
            m = self.memory_encoder(m_inputs).detach().cpu().numpy() 
            memory_index = faiss.IndexFlatL2(m.shape[1])
            memory_index.add(m)
            Ds, Is = memory_index.search(q, k)
            for index_list in Is:
                retrieved_top_ids = [all_candidaites[ind] for ind in index_list]
                top_k_accs.append(len([x for x in retrieved_top_ids if x in positive_ids])/k)

            del memory_index
        if seen_query_ids is not None:
            coverage = len(tested_query_ids & seen_query_ids)/len(tested_query_ids)
            self.logger.info(f"#tested_query_ids={len(tested_query_ids)}; coverage={coverage}")
        if seen_memory_ids is not None:
            coverage = len(tested_memory_ids & seen_memory_ids)/len(tested_memory_ids)
            self.logger.info(f"#tested_memory_ids={len(tested_memory_ids)}; coverage={coverage}")
        # self.logger.info(f"top_k_accs = {top_k_accs}; ")
        return np.mean(top_k_accs)

    def save_biencoder(self, query_encoder_path=None, memory_encoder_path=None):
        if not query_encoder_path:
            query_encoder_path = self.train_args.query_encoder_path
        if not memory_encoder_path:
            memory_encoder_path = self.train_args.memory_encoder_path

        def save_module(module, path):
            model_state_dict = {k: v.cpu() for (
                k, v) in module.state_dict().items()}
            torch.save(model_state_dict, path)
            self.logger.info(f"Model saved to {path}.")

        save_module(self.query_encoder, query_encoder_path)
        save_module(self.memory_encoder, memory_encoder_path)



    def get_query_representation(self, query_examples):
        """Using the concatenation"""
        before_all_vectors = get_bart_dual_representation(cl_trainer=self.cl_utils, 
                                                    bart_model=self.before_model, 
                                                    tokenizer=self.tokenizer, 
                                                    data_args=self.data_args, 
                                                    examples=query_examples)
        after_all_vectors = get_bart_dual_representation(cl_trainer=self.cl_utils, 
                                                    bart_model=self.after_model, 
                                                    tokenizer=self.tokenizer, 
                                                    data_args=self.data_args, 
                                                    examples=query_examples)
        bart_reps = []
        for b, a in zip(before_all_vectors, after_all_vectors):
            bart_reps.append(list(b)+list(a))
        bart_reps = np.array(bart_reps)
        self.query_encoder.eval()
        all_vectors = self.query_encoder(torch.Tensor(bart_reps)).detach().numpy()
        return all_vectors


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_dir_path",
                        default="exp_results/supervision_data/1020_dm_simple/")
    parser.add_argument("--num_ds_train_file", type=int, default=24)
    parser.add_argument("--num_ds_dev_file", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--run_mode", type=str, default="train")    # TODO:

    parser.add_argument("--query_encoder_path", type=str,
                        default="exp_results/supervision_data/$prefix.qry_encoder.pt")
    parser.add_argument("--memory_encoder_path", type=str,
                        default="exp_results/supervision_data/$prefix.mem_encoder.pt")
    parser.add_argument("--memory_index_path", type=str,
                        default="exp_results/supervision_data/$prefix.memory.index")
    parser.add_argument("--train_args_path", type=str,
                        default="exp_results/supervision_data/$prefix.train_args.json")

    # train_args

    parser.add_argument("--query_input_dim", type=int, default=768*2*2)
    parser.add_argument("--memory_input_dim", type=int, default=768*2)
    parser.add_argument("--hidden_dim", type=int, default=-1) # -1 means no hidden layer; 256 for example
    parser.add_argument("--dim_vector", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_steps", type=int, default=8000)
    parser.add_argument("--eval_per_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--qry_size", type=int, default=8)  # 1-16
    parser.add_argument("--pos_size", type=int, default=16)  # 1-8
    parser.add_argument("--neg_size", type=int, default=1)  # 1-8
    parser.add_argument("--patience", type=int, default=8)  
    parser.add_argument("--droprate", type=float, default=0)

    parser.add_argument('--use_query_mean', default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--run_name', default="1020_dm_simple", type=str)
    parser.add_argument('--save_ckpt', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--use_cuda', default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--wandb', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--query_only_after', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--query_only_before', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--query_delta', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))

    
    return parser


if __name__ == '__main__':

    biencoder_args = get_parser().parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    

    if biencoder_args.wandb and biencoder_args.run_mode == "train":
        run = wandb.init(reinit=True, project="FAIR_Biencoder", settings=wandb.Settings(start_method="fork"))
        run_name = wandb.run.name
        biencoder_args.run_name = run_name
        
        logger.info(f"run_name = {run_name}")

    biencoder_args.query_encoder_path = biencoder_args.query_encoder_path.replace("$prefix", biencoder_args.run_name)
    biencoder_args.memory_encoder_path = biencoder_args.memory_encoder_path.replace("$prefix", biencoder_args.run_name)
    biencoder_args.memory_index_path = biencoder_args.memory_index_path.replace("$prefix", biencoder_args.run_name)
    biencoder_args.train_args_path = biencoder_args.train_args_path.replace("$prefix", biencoder_args.run_name)


    set_seeds(biencoder_args.seed)

    if biencoder_args.run_mode == "train":
        with open(biencoder_args.train_args_path, "w") as f:
            json.dump(vars(biencoder_args), f)
        if biencoder_args.wandb:
            wandb.config.update(biencoder_args)
        if biencoder_args.query_only_after or biencoder_args.query_only_before:
            #  or biencoder_args.query_delta
            biencoder_args.query_input_dim = biencoder_args.query_input_dim // 2

        train_data, train_files = load_distant_supervision(
            biencoder_args.ds_dir_path, sample_size=biencoder_args.num_ds_train_file, logger=logger, train_args=biencoder_args)

        logger.info(f"num_groups = {len(train_data)}")

        eval_data, eval_files = load_distant_supervision(
            biencoder_args.ds_dir_path, sample_size=biencoder_args.num_ds_dev_file, logger=logger, exclude_files=train_files, train_args=biencoder_args)

        biencoder_memory_module = BiEncoderIndexManager(logger)
        biencoder_memory_module.train_args = biencoder_args
        biencoder_memory_module.init_biencoder_modules()
        biencoder_memory_module.train_biencoder(train_data, eval_data)
        if biencoder_args.save_ckpt:
            biencoder_memory_module.save_biencoder(
                biencoder_args.query_encoder_path, biencoder_args.memory_encoder_path)
        run.finish()

    elif biencoder_args.run_mode == "index": 
        # json.dump(vars(biencoder_args), open(biencoder_args.train_args_path, "w"))
        with open(biencoder_args.train_args_path, "r") as f:
            backup_args = json.load(f)
        biencoder_args.hidden_dim = backup_args["hidden_dim"]
        biencoder_args.query_input_dim = backup_args["query_input_dim"]
        biencoder_args.memory_input_dim = backup_args["memory_input_dim"]
        biencoder_args.hidden_dim = backup_args["hidden_dim"]
        biencoder_args.dim_vector = backup_args["dim_vector"]
        biencoder_args.use_query_mean = backup_args["use_query_mean"]
        

        from cmr.debug_algs import run_lifelong_finetune
        parser = run_lifelong_finetune.get_cli_parser()
        cl_args = parser.parse_args("")

        debugging_alg, data_args, base_model_args, debugger_args, logger = run_lifelong_finetune.setup_args(
            cl_args)
        cl_args.predict_batch_size = 8

        index_manager = BiEncoderIndexManager(logger)
        index_manager.train_args = biencoder_args
        index_manager.set_up_data_args(cl_args)
        index_manager.load_encoder_model(
            base_model_args, biencoder_args.memory_encoder_path, biencoder_args.query_encoder_path)

            

        index_manager.initial_memory_path = "data/mrqa_naturalquestions/mrqa_naturalquestions_train.jsonl"
        index_manager.set_up_initial_memory(index_manager.initial_memory_path)
        index_manager.save_memory_to_path("exp_results/data_streams/1021_biencoder_init_memory.pkl")


