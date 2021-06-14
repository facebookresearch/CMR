
from argparse import Namespace
from semanticdebugger.debug_algs.continual_finetune_alg import ContinualFinetuning
import logging
import os


def run():
    log_filename = "/tmp/debug.log"

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_filename),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    debugging_alg = ContinualFinetuning(logger=logger)

    bug_data_args = Namespace(
        bug_stream_json_path="bug_data/mrqa_naturalquestions_dev.static_bug_stream.json",
        do_lowercase=True,
        append_another_bos=True,
        max_input_length=888,
        max_output_length=30,
        task_name="mrqa_naturalquestions",
        train_batch_size=10,
        predict_batch_size=10,
        num_beams=3,
    )

    base_model_args = Namespace(
        model_type="facebook/bart-base",
        base_model_path="out/mrqa_naturalquestions_bart-base/best-model.pt"
    )

    debugger_args = Namespace(
        weight_decay=0.01,
        learning_rate=1e-5,
        adam_epsilon=1e-8,
        warmup_steps=0,
        total_steps=10000,
        num_epochs=3,
        gradient_accumulation_steps=1,
        max_grad_norm=0.1
    )

    debugging_alg.load_bug_streams(bug_data_args)
    debugging_alg.load_base_model(base_model_args)
    debugging_alg.debugger_setup(debugger_args)
    debugging_alg.online_debug()
    return


if __name__ == '__main__':
    run()
