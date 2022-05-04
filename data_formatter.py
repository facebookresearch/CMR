# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from enum import unique
from posixpath import split
from re import L
import datasets
import numpy as np
import os
import gzip
import sys
import json


def show_statistics(lines):
    len_list = []
    for l in lines:
        item = json.loads(l) 
        len_list.append(len(item["input"].split()))
    print(np.min(len_list), np.max(len_list),
          np.mean(len_list), np.median(len_list))
    return


def escape(s):
    # TODO: remove the markups
    filter_words = ["</P>", "<P>", "<Li>", "</Li>", "<Ul>", "</Ul>", "[DOC]", "[TLE]", "[PAR]", "[SEP]", "\n", "\t"]
    for fw in filter_words:
        s = s.replace(fw, "")
    return s.strip()

# Filtering the bad examples.


def example_pass(context):
    if "<table>" in context.lower() or "<td>" in context.lower():
        return False
    else:
        return True


def add_qmark(s):
    return s if s.endswith("?") else s + " ?"


def write_to_jsonl(lst, out_file):
    with open(out_file, "w") as fout:
        fout.write("\n".join(lst))
        # for line in lst:
        #     fout.write("{}\t{}\n".format(line[0], line[1]))


def deduplicate(lines):
    seen_inputs = set()
    unique_lines = []
    for line in lines:
        # print(line)
        item = json.loads(line)
        if item['input'] not in seen_inputs:
            unique_lines.append(line)
            seen_inputs.add(f"{item['input']}")
    # result = list(set(lines))
    print("deduplicate", len(lines), len(unique_lines))
    return unique_lines


class TextToTextDataset():

    def get_all_lines(self, dataset):
        train_lines = deduplicate(self.map_to_list(dataset, "train"))
        val_lines = deduplicate(self.map_to_list(dataset, "validation"))
        test_lines = deduplicate(self.map_to_list(dataset, "test"))

        show_statistics(train_lines)
        show_statistics(val_lines)

        # TODO: de-duplicate the lines!
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
        os.makedirs(os.path.join(path, self.task_identifier), exist_ok=True)
        prefix = os.path.join(path, self.task_identifier,
                              "{}".format(self.task_identifier))
        write_to_jsonl(train_lines, prefix + "_train.jsonl")
        write_to_jsonl(val_lines, prefix + "_dev.jsonl")
        if test_lines:
            write_to_jsonl(test_lines, prefix + "_test.jsonl")


class MRQA(TextToTextDataset):

    def __init__(self, task_identifier="mrqa", subset="SQuAD", mrqa_path="data/mrqa"):
        self.task_identifier = task_identifier + "_" + subset.lower()
        self.mrqa_path = mrqa_path
        self.subset = subset

    def map_to_list(self, dataset, split_name):
        if split_name not in dataset:
            print("No such split_name:", split_name)
            return []
        lines = []
        for datapoint in dataset[split_name]:
            if not datapoint["answers"]:
                print("empty answer")
                continue
            if not example_pass(datapoint["context"]):
                continue
            # add question mark!
            # lines.append(("Context: " + escape(datapoint["context"]) +
            #               " | Question: " +
            #               add_qmark(escape(datapoint["question"])),
            #               "\t".join([escape(a) for a in datapoint["answers"]])))

            # _input = "Context: " + escape(datapoint["context"]) + \
            #     " | " + "Question: " + add_qmark(escape(datapoint["question"]))
            # TODO: need re-training 
            _input = f'Question: {add_qmark(escape(datapoint["question"]))} </s> Context: {escape(datapoint["context"])}'
            _output = [escape(a) for a in datapoint["answers"]]
            _id = f"{self.task_identifier}-{split_name}-{len(lines)}"
            instance = {"id": _id, "input": _input, "output": _output}
            lines.append(json.dumps(instance))
        print("Three examples: \n" + "\n".join([str(_) for _ in lines[:3]]))
        return lines

    def load_dataset(self):
        def load_jsonl_gz(gzpath):
            data = []
            with gzip.open(gzpath, 'rb') as myzip:
                for example in myzip:
                    json_line = json.loads(example)
                    if "header" in json_line:
                        print(json_line["header"])
                        pass
                    else:
                        context = json_line["context"]
                        qa_items = []
                        for item in json_line["qas"]:
                            qa_items.append(dict(context=context,
                                                 qid=item["qid"],
                                                 question=item["question"],
                                                 answers=list(set(item["answers"]))))
                        data.extend(qa_items)
            return data

        path_to_train = os.path.join(
            self.mrqa_path, "mrqa_train", self.subset+".jsonl.gz")
        path_to_dev = os.path.join(
            self.mrqa_path, "mrqa_dev", self.subset+".jsonl.gz")
        dataset = {}
        dataset["train"] = load_jsonl_gz(path_to_train)
        dataset["validation"] = load_jsonl_gz(path_to_dev)
        return dataset


class NLI(TextToTextDataset):

    def __init__(self, task_identifier="snli"): 
        self.task_identifier = task_identifier
        # for classification tasks, specify the meaning of each label
        self.prompt = " "  # are two sentences entailment or not entailment?
        if self.task_identifier in ["snli", "anli", "multi_nli"]:
            self.label = {
                0: ["entailment"],     
                1: ["neutral"],   
                2: ["contradiction"]
            }
        elif self.task_identifier == "qnli":
            self.label = {
                0: ["entailment"],     
                1: ["neutral"], 
            }
        elif self.task_identifier == "scitail":
            self.label = {
                "entails": ["entailment"],     
                "neutral": ["neutral"],    
            }

    def get_all_lines(self, dataset, splits=["train", "validation", "test"]):
        all_lines = {}
        for split in splits:
            all_lines[split] = deduplicate(self.map_to_list(dataset, split)) 
            show_statistics(all_lines[split]) 

        # TODO: de-duplicate the lines!
        return all_lines

    def write_dataset(self, path):
        """
        return train, dev, test
        """

        # load dataset
        dataset = self.load_dataset()

        # formulate into list (for consistency in np.random)
        if self.task_identifier in ["snli", "scitail", "qnli"]:
            splits = ["train", "validation", "test"]
        elif self.task_identifier == "anli":
            splits = ['train_r1', 'dev_r1', 'test_r1', 'train_r2', 'dev_r2', 'test_r2', 'train_r3', 'dev_r3', 'test_r3']
        elif self.task_identifier == "multi_nli":
            splits = ['validation_matched', 'validation_mismatched'] 
            
        all_lines = self.get_all_lines(dataset, splits)

        # shuffle the data
        # np.random.seed(seed)
        # np.random.shuffle(train_lines)
        os.makedirs(os.path.join(path, self.task_identifier), exist_ok=True)
        prefix = os.path.join(path, self.task_identifier,
                              "{}".format(self.task_identifier))
        for split in splits:
            write_to_jsonl(all_lines[split], f"{prefix}_{split}.jsonl") 

    def map_to_list(self, dataset, split_name):
        lines = []
        for datapoint in dataset[split_name]:
            # print(datapoint["label"])
            if datapoint["label"] not in self.label:
                continue
            # lines.append(("Premise: " + datapoint["premise"] + " | Hypothesis: " +
            #              datapoint["hypothesis"], self.label[datapoint["label"]]))
            _id = f"{self.task_identifier}-{split_name}-{len(lines)}"
            if self.task_identifier == "qnli":
                _input = f'Premise: {datapoint["sentence"]} </s> Hypothesis: {datapoint["question"]}'
            else:
                _input = f'Premise: {datapoint["premise"]} </s> Hypothesis: {datapoint["hypothesis"]}'
            _input += " | Options: entailment, neutral, contradiction "
            _output = self.label[datapoint["label"]]
            instance = {"id": _id, "input": _input, "output": _output}
            lines.append(json.dumps(instance))
        print("Three examples: \n" + "\n".join([str(_) for _ in lines[:3]]))
        return lines

    def load_dataset(self):
        if self.task_identifier == "scitail":
            return datasets.load_dataset("scitail", "dgem_format")
        elif self.task_identifier == "qnli":
            return datasets.load_dataset("glue", "qnli")
        else:
            return datasets.load_dataset(self.task_identifier)




class CSR(TextToTextDataset):

    def __init__(self, task_identifier="commonsense_qa"): 
        self.task_identifier = task_identifier
        # for classification tasks, specify the meaning of each label
        # self.prompt = " "  # are two sentences entailment or not entailment?
        # if self.task_identifier in ["snli", "anli", "multi_nli"]:
        #     self.label = {
        #         0: ["entailment"],     
        #         1: ["neutral"],   
        #         2: ["contradiction"]
        #     }
        # elif self.task_identifier == "qnli":
        #     self.label = {
        #         0: ["entailment"],     
        #         1: ["neutral"], 
        #     }
        # elif self.task_identifier == "scitail":
        #     self.label = {
        #         "entails": ["entailment"],     
        #         "neutral": ["neutral"],    
        #     }

    def get_all_lines(self, dataset, splits=["train", "validation", "test"]):
        all_lines = {}
        for split in splits:
            all_lines[split] = deduplicate(self.map_to_list(dataset, split)) 
            show_statistics(all_lines[split]) 

        # TODO: de-duplicate the lines!
        return all_lines

    def write_dataset(self, path):
        """
        return train, dev, test
        """

        # load dataset
        dataset = self.load_dataset()

        # formulate into list (for consistency in np.random)
        # if self.task_identifier in ["snli", "scitail", "qnli"]:
        #     splits = ["train", "validation", "test"]
        # elif self.task_identifier == "anli":
        #     splits = ['train_r1', 'dev_r1', 'test_r1', 'train_r2', 'dev_r2', 'test_r2', 'train_r3', 'dev_r3', 'test_r3']
        # elif self.task_identifier == "multi_nli":
        #     splits = ['validation_matched', 'validation_mismatched'] 
        splits = ["train", "validation"]
        all_lines = self.get_all_lines(dataset, splits)

        # shuffle the data
        # np.random.seed(seed)
        # np.random.shuffle(train_lines)
        os.makedirs(os.path.join(path, self.task_identifier), exist_ok=True)
        prefix = os.path.join(path, self.task_identifier,
                              "{}".format(self.task_identifier))
        for split in splits:
            write_to_jsonl(all_lines[split], f"{prefix}_{split}.jsonl") 

    def map_to_list(self, dataset, split_name):
        lines = []
        for datapoint in dataset[split_name]: 
            choices = datapoint["choices"]
            choices_map = {}
            choice_strs = []
            for ind, (key, choice) in enumerate(list(zip(choices["label"], choices["text"]))):
                if self.task_identifier == "openbookqa":
                    key = list("ABCDEF")[ind]  
                choices_map[key]  = choice
                choice_strs.append(f"{key}: {choice}")
             
            _id = f"{self.task_identifier}-{split_name}-{len(lines)}"
            
            if self.task_identifier == "openbookqa":
                _input = f'Question: {datapoint["question_stem"]} </s> {" | ".join(choice_strs)}'
            else:
                _input = f'Question: {datapoint["question"]} </s> {" | ".join(choice_strs)}'

            _output = [choices_map[datapoint["answerKey"]]]
            instance = {"id": _id, "input": _input, "output": _output}
            lines.append(json.dumps(instance))
        print("Three examples: \n" + "\n".join([str(_) for _ in lines[:3]]))
        return lines

    def load_dataset(self): 
        if self.task_identifier == "ai2_arc-easy":
            return datasets.load_dataset("ai2_arc", "ARC-Easy")    
        elif self.task_identifier == "ai2_arc-hard":
            return datasets.load_dataset("ai2_arc", "ARC-Challenge")    
        elif self.task_identifier == "openbookqa":
            return datasets.load_dataset("openbookqa", "main")    
        return datasets.load_dataset(self.task_identifier)




def format(dataset_name, path="./"):
    print("Formatting ", dataset_name)
    if dataset_name.startswith("mrqa_"):
        data = MRQA(subset=dataset_name.split("_")[1])
        data.write_dataset(path)
    elif dataset_name.startswith("nli#"):
        name = dataset_name.split("#")[1]
        data = NLI(task_identifier=name)
        data.write_dataset(path)
    elif dataset_name.startswith("csr#"):
        name = dataset_name.split("#")[1]
        data = CSR(task_identifier=name)
        data.write_dataset(path)


path = "data/"
if len(sys.argv) >= 2:
    path = sys.argv[1]


format("mrqa_SQuAD", path)
format("mrqa_TriviaQA", path)
format("mrqa_NaturalQuestions", path)
format("mrqa_HotpotQA", path)
format("mrqa_NewsQA", path)
format("mrqa_SearchQA", path)


# format("nli#snli", path)
# format("nli#anli", path)
# format("nli#multi_nli", path)
# format("nli#scitail", path)


# format("csr#commonsense_qa", path)
# format("csr#riddle_sense", path)
# format("csr#ai2_arc-easy", path)
# format("csr#ai2_arc-hard", path)
# format("csr#openbookqa", path)
 
