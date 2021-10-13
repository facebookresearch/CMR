import argparse
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
import json

from semanticdebugger.models.utils import set_seeds


def load_distant_supervision(folder_name, sample_size=1, logger=None, specified_names=None, exclude_files=[]):
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

    return ds_items, pkl_files


class MLP(Module):
    def __init__(self, input_dim, output_dim, hidden_dim, droprate=0.01):
        super().__init__()
        self.layers = torch.nn.Sequential(
            #   torch.nn.Flatten(),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Sigmoid(),
            nn.Dropout(droprate),
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


def create_batch_from_groups(groups, qry_size=1, pos_size=1, neg_size=1, seen_query_ids=None, seen_memory_ids=None):
    queries, candidates, targets = [], [], []
    for group in groups:
        # TODO: this is overly recorded..
        if seen_query_ids is not None:
            seen_query_ids.update(set(list(group["query"].keys())))
        if seen_memory_ids is not None:
            seen_memory_ids.update((set(list(group["positive"].keys()))))
            seen_memory_ids.update((set(list(group["negative"].keys()))))
        
        # queries.append(random.choice(list(group["query"].values())))
        
        queries += random.choices(list(group["query"].values()), k=qry_size)
        target = len(candidates)
        candidates.append(random.choice(list(group["positive"].values())))  # a single positive
        candidates += random.choices(list(group["negative"].values()), k=neg_size)
        targets += [target] * qry_size

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
        self.dim_vector = 256  # final dim
        self.memory_encoder = None
        self.query_encoder = None
        self.train_args = None

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
                                 self.hidden_dim*2, droprate=self.train_args.droprate)

    def get_representation(self, examples):
        """only for the memory encoding now"""
        bart_reps = super().get_representation(examples)
        bart_reps = np.array(bart_reps)
        self.memory_encoder.eval()
        all_vectors = self.memory_encoder(torch.Tensor(bart_reps)).detach().numpy()
        return all_vectors

    def train_biencoder(self, train_data, eval_data):
        trainable_params = list(self.query_encoder.parameters()) + \
            list(self.memory_encoder.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.train_args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.1, last_epoch=-1)
        gradient_acc_steps = 1

        seen_query_ids = set()
        seen_memory_ids = set()
        best_eval_acc = self.eval_func(
            eval_data, k=8, seen_query_ids=seen_query_ids, seen_memory_ids=seen_memory_ids)
        self.logger.info(f"Zero-Training Evaluation Top-k acc: {best_eval_acc}")

        losses = []
        for _step in tqdm(range(self.train_args.n_steps), desc="Training steps"):
            self.memory_encoder.train()
            self.query_encoder.train()
            # self.logger.info(f"Training step {_step}/{self.train_args.n_steps}")

            sampled_groups = random.choices(train_data, k=self.train_args.batch_size)
            queries, candidates, targets = create_batch_from_groups(
                sampled_groups, qry_size=self.train_args.qry_size, neg_size=self.train_args.neg_size, seen_query_ids=seen_query_ids, seen_memory_ids=seen_memory_ids)
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
            self.logger.info(
                f"---- Completed epoch with avg training loss {sum(losses)/len(losses)}.")
            # self.logger.info(f"self.query_encoder.layers[0].weight = {self.query_encoder.layers[0].weight}")
            # self.logger.info(f"self.memory_encoder.layers[0].weight = {self.memory_encoder.layers[0].weight}")
            if _step % self.train_args.eval_per_steps == 0:
                eval_acc = self.eval_func(
                    eval_data, k=8, seen_query_ids=seen_query_ids, seen_memory_ids=seen_memory_ids)
                best_eval_acc = max(best_eval_acc, eval_acc)
                self.logger.info(
                    f"Evaluation Top-8 acc @ {_step}: {eval_acc} | best_eval_acc={best_eval_acc}")

    def eval_func(self, eval_data, k=5, seen_query_ids=None, seen_memory_ids=None, filter=False):
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
                if filter and seen_memory_ids is not None and ex_id in seen_memory_ids:
                    # Remove the seen memory ids
                    continue
                positive_ids.add(ex_id)
                all_candidaites.append(ex_id)
                tested_memory_ids.add(ex_id)
                all_candidate_vectors.append(vector)
            for ex_id, vector in item["negative"].items():
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
        if seen_memory_ids is not None:
            coverage = len(tested_memory_ids & seen_memory_ids)/len(tested_memory_ids)
            print(f"#tested_memory_ids={len(tested_memory_ids)}; coverage={coverage}")
        # print(f"top_k_accs = {top_k_accs}; ")
        return np.mean(top_k_accs)

    def save_biencoder(self, query_encoder_path, memory_encoder_path):
        def save_module(module, path):
            model_state_dict = {k: v.cpu() for (
                k, v) in module.state_dict().items()}
            torch.save(model_state_dict, path)
            self.logger.info(f"Model saved to {path}.")

        save_module(self.query_encoder, query_encoder_path)
        save_module(self.memory_encoder, memory_encoder_path)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_dir_path",
                        default="exp_results/supervision_data/1012_dm_simple/")
    parser.add_argument("--num_ds_train_file", type=int, default=100)
    parser.add_argument("--num_ds_dev_file", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--run_mode", type=str, default="index")    # TODO:

    parser.add_argument("--query_encoder_path", type=str,
                        default="exp_results/supervision_data/1012_dm_simple.qry_encoder.pt")
    parser.add_argument("--memory_encoder_path", type=str,
                        default="exp_results/supervision_data/1012_dm_simple.mem_encoder.pt")
    parser.add_argument("--memory_index_path", type=str,
                        default="exp_results/supervision_data/1012_dm_simple.memory.index")
    parser.add_argument("--train_args_path", type=str,
                        default="exp_results/supervision_data/1012_dm_simple.train_args.json")

    # train_args

    parser.add_argument("--query_input_dim", type=int, default=768*2*2)
    parser.add_argument("--memory_input_dim", type=int, default=768*2)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_vector", type=int, default=256)

    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--n_steps", type=int, default=300)
    parser.add_argument("--eval_per_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--qry_size", type=int, default=8)
    parser.add_argument("--neg_size", type=int, default=8)
    parser.add_argument("--droprate", type=float, default=0.1)
    return parser


if __name__ == '__main__':

    biencoder_args = get_parser().parse_args()
    json.dump(vars(biencoder_args), open(biencoder_args.train_args_path, "w"))

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    set_seeds(biencoder_args.seed)

    if biencoder_args.run_mode == "train":
        train_data, train_files = load_distant_supervision(
            biencoder_args.ds_dir_path, sample_size=biencoder_args.num_ds_train_file, logger=logger)

        logger.info(f"num_groups = {len(train_data)}")

        eval_data, eval_files = load_distant_supervision(
            biencoder_args.ds_dir_path, sample_size=biencoder_args.num_ds_dev_file, logger=logger, exclude_files=train_files)

        biencoder_memory_module = BiEncoderIndexManager(logger)
        biencoder_memory_module.train_args = biencoder_args
        biencoder_memory_module.init_biencoder_modules()
        biencoder_memory_module.train_biencoder(train_data, eval_data)
        biencoder_memory_module.save_biencoder(
            biencoder_args.query_encoder_path, biencoder_args.memory_encoder_path)

    elif biencoder_args.run_mode == "index":

        from semanticdebugger.debug_algs import run_lifelong_finetune
        parser = run_lifelong_finetune.get_cli_parser()
        cl_args = parser.parse_args()

        debugging_alg, data_args, base_model_args, debugger_args, logger = run_lifelong_finetune.setup_args(
            cl_args)
        cl_args.predict_batch_size = 8

        index_manager = BiEncoderIndexManager(logger)
        index_manager.train_args = biencoder_args
        index_manager.set_up_data_args(cl_args)
        index_manager.load_encoder_model(
            base_model_args, biencoder_args.memory_encoder_path, biencoder_args.query_encoder_path)

            

        index_manager.initial_memory_path = "exp_results/data_streams/mrqa.nq_train.memory.jsonl"
        index_manager.set_up_initial_memory(index_manager.initial_memory_path)
        index_manager.save_memory_to_path("exp_results/data_streams/1012_biencoder_init_memory.pkl")
