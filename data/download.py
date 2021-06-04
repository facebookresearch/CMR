from re import L
import datasets
import numpy as np
import os

import sys 

def escape(s):
    return s.replace("\n", " ").replace("\t", " ").strip()

def add_qmark(s):
    return s if s.endswith("?") else s + " ?"

def write_to_tsv(lst, out_file):
    with open(out_file, "w") as fout:
        for line in lst:
            fout.write("{}\t{}\n".format(line[0], line[1]))

class TextToTextDataset():

    def get_all_lines(self, dataset):
        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        val_lines = self.map_hf_dataset_to_list(dataset, "validation")
        test_lines = self.map_hf_dataset_to_list(dataset, "test")
        return train_lines, val_lines, test_lines

    
    def write_dataset(self, path):
        """
        return train, dev, test
        """

        # load dataset
        dataset = self.load_dataset()

        # formulate into list (for consistency in np.random)
        train_lines, val_lines, test_lines = self.get_all_lines(dataset)

        # shuffle the data
        # np.random.seed(seed)
        # np.random.shuffle(train_lines) 
        os.makedirs(os.path.join(path, self.hf_identifier), exist_ok=True)
        prefix = os.path.join(path, self.hf_identifier, "{}".format(self.hf_identifier))
        write_to_tsv(train_lines, prefix + "_train.tsv")
        write_to_tsv(val_lines, prefix + "_dev.tsv")
        write_to_tsv(test_lines, prefix + "_test.tsv") 

class Kilt_NQ(TextToTextDataset):

    def __init__(self, hf_identifier="kilt_nq"):
        self.hf_identifier = hf_identifier

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append((escape(add_qmark(datapoint["input"])), "\t".join([escape(item["answer"]) for item in datapoint["output"]])))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('kilt_tasks','nq')

class Kilt_TriviaQA(TextToTextDataset):

    def __init__(self, hf_identifier="kilt_triviaqa"):
        self.hf_identifier = hf_identifier

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append((escape(add_qmark(datapoint["input"])), "\t".join([escape(item["answer"]) for item in datapoint["output"]])))
        return lines

    def load_dataset(self):
         # Get the KILT task datasets
        kilt_triviaqa = datasets.load_dataset("kilt_tasks", name="triviaqa_support_only")

        # Most tasks in KILT already have all required data, but KILT-TriviaQA
        # only provides the question IDs, not the questions themselves.
        # Thankfully, we can get the original TriviaQA data with:
        trivia_qa = datasets.load_dataset('trivia_qa', 'unfiltered.nocontext')

        # The KILT IDs can then be mapped to the TriviaQA questions with: triviaqa_map
        for k in ['train', 'validation', 'test']:
            triviaqa_map = dict([(q_id, i) for i, q_id in enumerate(trivia_qa[k]['question_id'])])
            kilt_triviaqa[k] = kilt_triviaqa[k].filter(lambda x: x['id'] in triviaqa_map)
            kilt_triviaqa[k] = kilt_triviaqa[k].map(lambda x: {'input': trivia_qa[k][triviaqa_map[x['id']]]['question']})

        return kilt_triviaqa


class Glue_QNLI(TextToTextDataset):

    def __init__(self, hf_identifier="glue_qnli"):
        self.hf_identifier = hf_identifier
        # for classification tasks, specify the meaning of each label
        self.prompt = " " # are two sentences entailment or not entailment?
        self.label = {
            0: "entailment",    # entailment
            1: "irrelevant",  # not_entailment
            -1: "unkown"
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            lines.append(("Context: " + datapoint["sentence"] + " | Question: " + datapoint["question"], self.label[datapoint["label"]]))
        print("Three examples: \n"+ "\n".join([str(_) for _ in lines[:3]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('glue','qnli')


def download(dataset_name, path="./"):
    print("Downloading ", dataset_name)
    if dataset_name == "kilt_nq":
        data = Kilt_NQ()
        data.write_dataset(path)
    elif dataset_name == "kilt_triviaqa":
        data = Kilt_TriviaQA()
        data.write_dataset(path)   
    elif dataset_name == "glue_qnli":
        data = Glue_QNLI()
        data.write_dataset(path)   

path = "."
if len(sys.argv) >=2:
    path = sys.argv[1] 
download("kilt_nq", path)
# download("kilt_triviaqa", path)
# download("glue_qnli", path)
