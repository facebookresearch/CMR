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


def inference_api(config_file, test_file):
    parser = get_parser()
    args = parser.parse_args()
    # load config from json 
    
    logger = None 

    tokenizer = BartTokenizer.from_pretrained("bart-large")
    test_data = GeneralDataset(logger, args, args.dev_file, data_type="dev", is_training=False, task_name=args.dataset)
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


inference_api(config_file="scripts/infer_mrqa_bart_base.config", test_file="data/mrqa_naturalquestions/mrqa_naturalquestions_dev.tsv")