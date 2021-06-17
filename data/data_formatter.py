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


class Kilt_NQ(TextToTextDataset):

    def __init__(self, task_identifier="kilt_nq"):
        self.task_identifier = task_identifier

    def map_to_list(self, dataset, split_name):
        lines = []
        for datapoint in dataset[split_name]:
            lines.append((escape(add_qmark(datapoint["input"])), "\t".join(
                [escape(item["answer"]) for item in datapoint["output"]])))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('kilt_tasks', 'nq')


class Kilt_TriviaQA(TextToTextDataset):

    def __init__(self, task_identifier="kilt_triviaqa"):
        self.task_identifier = task_identifier

    def map_to_list(self, dataset, split_name):
        lines = []
        for datapoint in dataset[split_name]:
            lines.append((escape(add_qmark(datapoint["input"])), "\t".join(
                [escape(item["answer"]) for item in datapoint["output"]])))
        return lines

    def load_dataset(self):
        # Get the KILT task datasets
        kilt_triviaqa = datasets.load_dataset(
            "kilt_tasks", name="triviaqa_support_only")

        # Most tasks in KILT already have all required data, but KILT-TriviaQA
        # only provides the question IDs, not the questions themselves.
        # Thankfully, we can get the original TriviaQA data with:
        trivia_qa = datasets.load_dataset('trivia_qa', 'unfiltered.nocontext')

        # The KILT IDs can then be mapped to the TriviaQA questions with: triviaqa_map
        for k in ['train', 'validation', 'test']:
            triviaqa_map = dict(
                [(q_id, i) for i, q_id in enumerate(trivia_qa[k]['question_id'])])
            kilt_triviaqa[k] = kilt_triviaqa[k].filter(
                lambda x: x['id'] in triviaqa_map)
            kilt_triviaqa[k] = kilt_triviaqa[k].map(
                lambda x: {'input': trivia_qa[k][triviaqa_map[x['id']]]['question']})

        return kilt_triviaqa


class Glue_QNLI(TextToTextDataset):

    def __init__(self, task_identifier="glue_qnli"):
        self.task_identifier = task_identifier
        # for classification tasks, specify the meaning of each label
        self.prompt = " "  # are two sentences entailment or not entailment?
        self.label = {
            0: "entailment",    # entailment
            1: "irrelevant",  # not_entailment
            -1: "unkown"
        }

    def map_to_list(self, dataset, split_name):
        lines = []
        for datapoint in dataset[split_name]:
            lines.append(("Context: " + datapoint["sentence"] + " | Question: " +
                         datapoint["question"], self.label[datapoint["label"]]))
        print("Three examples: \n" + "\n".join([str(_) for _ in lines[:3]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset('glue', 'qnli')


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
            _input = "Context: " + escape(datapoint["context"]) + \
                " | " + "Question: " + add_qmark(escape(datapoint["question"]))
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


def format(dataset_name, path="./"):
    print("Formatting ", dataset_name)
    if dataset_name == "kilt_nq":
        data = Kilt_NQ()
        data.write_dataset(path)
    elif dataset_name == "kilt_triviaqa":
        data = Kilt_TriviaQA()
        data.write_dataset(path)
    elif dataset_name == "glue_qnli":
        data = Glue_QNLI()
        data.write_dataset(path)
    elif dataset_name.startswith("mrqa_"):
        data = MRQA(subset=dataset_name.split("_")[1])
        data.write_dataset(path)


path = "data/"
if len(sys.argv) >= 2:
    path = sys.argv[1]

# format("kilt_nq", path)
# format("kilt_triviaqa", path)
# format("glue_qnli", path)

format("mrqa_SQuAD", path)
format("mrqa_TriviaQA", path)
format("mrqa_NaturalQuestions", path)
format("mrqa_HotpotQA", path)


# shuf -n 1000 dev_file data/${task}/${task}_dev.mini.jsonl
