# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import os
import sys
from argparse import Namespace

import torch
from cmr.models.mybart import MyBart
from cmr.models.run_bart import inference
from cmr.models.utils import (convert_model_to_single_gpu,
                                           freeze_embeds, trim_batch)
from cmr.task_manager.dataloader import GeneralDataset
from tqdm import tqdm
from transformers import BartConfig, BartTokenizer


def inference_api(config_file, test_file, logger, data_dist, num_shards, local_id):

    with open(config_file) as f:
        config_args = eval(f.read())  # an Namespace object in python language
    args = config_args
    logger.info(f"Config args: {config_args}")
    # load config from json

    test_data = GeneralDataset(
        logger, args, test_file, data_type="dev", is_training=False, task_name=args.dataset, data_dist=data_dist, num_shards=num_shards, local_id=local_id)
    tokenizer = BartTokenizer.from_pretrained("bart-large")
    test_data.load_dataset(tokenizer, skip_cache=data_dist)
    test_data.load_dataloader()

    checkpoint = os.path.join(args.predict_checkpoint)

    logger.info("Loading checkpoint from {} ....".format(checkpoint))
    model = MyBart.from_pretrained(args.model,
                                   state_dict=convert_model_to_single_gpu(torch.load(checkpoint)))
    logger.info("Loading checkpoint from {} .... Done!".format(checkpoint))
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    model.eval()

    predictions = inference(
        model, test_data, save_predictions=False, verbose=True, args=args, logger=logger, return_all=False, predictions_only=True)
    return predictions
    # logger.info("%s on %s data: %.s" % (test_data.metric, test_data.data_type, str(result)))

