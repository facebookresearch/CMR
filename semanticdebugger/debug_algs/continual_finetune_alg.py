from .common_utils import OnlineDebuggingMethod
from semanticdebugger.models.mybart import MyBart
from semanticdebugger.models.utils import freeze_embeds, trim_batch, convert_model_to_single_gpu
import torch
import numpy as np 
from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

class ContinualFinetuning(OnlineDebuggingMethod):
    def __init__(self):
        self.base_model = None 
        self.debugger = None 
        
    def load_base_model(self, model_type, base_model_path):
        self.base_model = MyBart.from_pretrained(model_type,
                                           state_dict=convert_model_to_single_gpu(torch.load(base_model_path)))
        pass


        