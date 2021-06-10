import sys
import os
from semanticdebugger.models.mybart import MyBart
from semanticdebugger.models.utils import freeze_embeds, trim_batch, convert_model_to_single_gpu
import json
import torch
from tqdm import tqdm
from transformers import BartTokenizer, BartConfig
from semanticdebugger.task_manager.dataloader import GeneralDataset
from semanticdebugger.models.run_bart import inference
from semanticdebugger.cli_bart import get_parser
from argparse import Namespace
import logging

def inference_api(config_file, test_file, logger):
    parser = get_parser()
    with open(config_file) as f:
        config_args = eval(f.read())  # an Namespace object in python language
    args = parser.parse_args(namespace=config_args) 
    # load config from json  
    
    test_data = GeneralDataset(logger, args, test_file, data_type="dev", is_training=False, task_name=args.dataset)
    tokenizer = BartTokenizer.from_pretrained("bart-large")
    test_data.load_dataset(tokenizer)
    test_data.load_dataloader()

    checkpoint = os.path.join(args.predict_checkpoint)

    logger.info("Loading checkpoint from {} ....".format(checkpoint))
    model = MyBart.from_pretrained(args.model,
                                state_dict=convert_model_to_single_gpu(torch.load(checkpoint)))
    logger.info("Loading checkpoint from {} .... Done!".format(checkpoint))
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    model.eval()

    test_performance = inference(model, test_data, save_predictions=True, verbose=True, args=args, logger=logger)
    logger.info("%s on %s data: %.s" % (test_data.metric, test_data.data_type, str(test_performance)))

