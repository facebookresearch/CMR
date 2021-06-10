import os
import json
from .base_datamanager import MyQADataset, MyDataLoader
from .eval_metrics import METRICS, evaluate_func

class GeneralDataset(object):

    def __init__(self, logger, args, data_path, data_type, is_training, task_name):
        # should give the tasks used in this split in the var "tasks"
        self.data_path = data_path
        self.data_type = data_type
        
        self.data = []
        self.task_name = task_name
        with open(data_path) as fin:
            lines = fin.readlines()

        # train_examples = []
        for line in lines:
            d = line.strip().split("\t")
            self.data.append((d[0], d[1:]))
            

        self.is_training = is_training
        self.load = not args.debug
        self.logger = logger
        self.args = args

        self.metric = METRICS[self.task_name]
        # self.max_input_length = self.args.max_input_length
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None

        self.gen_early_stop = False

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(".tsv", "-{}.json".format(postfix)))
        
        
        if self.load and os.path.exists(preprocessed_path):
            # load preprocessed input
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                    metadata = json.load(f)

        else:
            self.logger.info("Start tokenizing ... {} instances".format(len(self.data)))

            inputs = []
            outputs = []

            for dp in self.data:
                # Add the task name to the input
                # inputs.append(" [{}] {}".format(self.task_name, dp[0]))
                inputs.append(dp[0])
                outputs.append(dp[1]) # is a list

            self.logger.info("Printing 3 examples")
            for i in range(3):
                self.logger.info(inputs[i])
                self.logger.info(outputs[i])

            outputs, metadata = self.flatten(outputs) # what is metadata?
            self.logger.info("Printing 3 examples's outputs and metadata after flattening")
            for i in range(3):
                self.logger.info(outputs[i])
                self.logger.info(metadata[i])

            if self.args.do_lowercase:
                inputs = [input0.lower() for input0 in inputs]
                outputs = [output0.lower() for output0 in outputs]
            if self.args.append_another_bos:
                inputs = ["<s> "+input0 for input0 in inputs]
                outputs = ["<s> " +output0 for output0 in outputs]
            
            self.logger.info("Tokenizing Input ...")
            tokenized_input = tokenizer.batch_encode_plus(inputs,
                                                         pad_to_max_length=True,
                                                         max_length=self.args.max_input_length)
            self.logger.info("Tokenizing Input ... Done!")
            self.logger.info("Tokenizing Output ...")
            tokenized_output = tokenizer.batch_encode_plus(outputs,
                                                       pad_to_max_length=True,
                                                       max_length=self.args.max_output_length)
            self.logger.info("Tokenizing Output ... Done!")
            input_ids, attention_mask = tokenized_input["input_ids"], tokenized_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = tokenized_output["input_ids"], tokenized_output["attention_mask"]
            if self.load: 
                preprocessed_data = [input_ids, attention_mask,
                                     decoder_input_ids, decoder_attention_mask,
                                     metadata]
                self.logger.info("Save preprocessed data ...")
                with open(preprocessed_path, "w") as f:
                    json.dump([input_ids, attention_mask,
                               decoder_input_ids, decoder_attention_mask,
                               metadata], f)
                self.logger.info("Save preprocessed data ... Done!")
        self.logger.info("len(input_ids): {}".format(len(input_ids)))
        self.logger.info("len(decoder_input_ids): {}".format(len(decoder_input_ids)))
        self.logger.info("len(attention_mask): {}".format(len(attention_mask)))
        self.logger.info("len(decoder_attention_mask): {}".format(len(decoder_attention_mask)))
        
        self.dataset = MyQADataset(input_ids, attention_mask,
                                        decoder_input_ids, decoder_attention_mask,
                                        in_metadata=None, out_metadata=metadata,
                                        is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions, verbose=False):
        assert len(predictions)==len(self), (len(predictions), len(self))
        predictions = [prediction.strip() for prediction in predictions]
        return evaluate_func(predictions, self.data, self.metric)
        # ems = []
        # for (prediction, dp) in zip(predictions, self.data):
        #     ems.append(get_exact_match(prediction.strip(), [dp[1]]))
        # return np.mean(ems)

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))

        predictions = ['n/a' if len(prediction.strip())==0 else prediction for prediction in predictions]
        prediction_text = [prediction.strip()+'\n' for prediction in predictions]
        save_path = os.path.join(self.args.output_dir, "{}_predictions.txt".format(self.args.prefix))
        with open(save_path, "w") as f:
            f.writelines(prediction_text)
        
        self.logger.info("Saved prediction in {}".format(save_path))
