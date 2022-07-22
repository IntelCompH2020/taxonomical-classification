# coding=utf-8

import os
import datasets
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

class TrainerHF(datasets.GeneratorBasedBuilder):

    def __init__(self, **kwargs):
        self.text_column     = kwargs["text_column"]
        self.label_column    = kwargs["label_column"]
        self.path_to_train_data = self.path_to_dev_data = self.path_to_test_data = None
        self.path_to_parquet = []
        test_data = False
        if "path_to_test_data" in kwargs and kwargs["path_to_test_data"] is not None:
            self.path_to_test_data = kwargs["path_to_test_data"]
            test_data = True
        if "path_to_dev_data" in kwargs and kwargs["path_to_dev_data"] is not None:
            self.path_to_dev_data = kwargs["path_to_dev_data"]
            self.path_to_parquet.append(self.path_to_dev_data)
        if "path_to_train_data" in kwargs and kwargs["path_to_train_data"] is not None:
            self.path_to_train_data = kwargs["path_to_train_data"]
            self.path_to_parquet.append(self.path_to_train_data)
        if self.path_to_parquet == [] and not test_data:
            raise Exception("No data paths were provided.")

        # Read the data
        self.classes = set()
        list_of_paths = []
        for parquet_split_path in self.path_to_parquet:
            if os.path.isfile(parquet_split_path):
                list_of_paths.append(parquet_split_path)
            else:
                list_of_paths += [os.path.join(root, file_) \
                        for root, _, files in os.walk(parquet_split_path) \
                        for file_ in files if file_.endswith(".parquet")]

        for file_ in tqdm(list_of_paths):
            labels_df = pq.read_table(file_, columns=[self.label_column]).to_pandas()
            for label in labels_df[self.label_column].explode().unique():
                self.classes.add(label)

        custom_columns = ["text_column", "label_column", "path_to_train_data", "path_to_dev_data", "path_to_test_data"]
        kwargs = {k:v for k,v in kwargs.items() if k not in custom_columns}
        super(TrainerHF, self).__init__(**kwargs)

    def _info(self):
        # Create features dict
        features_dict = {"text": datasets.Value("string")}
        # Add all labels as boolean features
        for label in self.classes:
            features_dict[label] = datasets.Value("bool")
        return datasets.DatasetInfo(
            features=datasets.Features(features_dict),
            description="",
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        paths = []
        if self.path_to_train_data is not None:
            paths.append(datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": self.path_to_train_data, "split": datasets.Split.TRAIN}))
        if self.path_to_dev_data is not None:
            paths.append(datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": self.path_to_dev_data, "split": datasets.Split.VALIDATION}))
        if self.path_to_test_data is not None:
            paths.append(datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": self.path_to_test_data, "split": datasets.Split.TEST}))
        return paths

    def _generate_examples(self, filepath, split):
        list_of_paths = []
        idx = 0
        if os.path.isfile(filepath):
            list_of_paths.append(filepath)
        else:
            list_of_paths += [os.path.join(root, file_) for root, _, files in os.walk(filepath) for file_ in files if file_.endswith(".parquet")]
        for file_ in list_of_paths:
            if split == datasets.Split.TEST:
                df = pq.read_table(file_, columns=[self.text_column]).to_pandas()
                for id_, row in df.iterrows():
                    features = {feature: False for feature in self.classes}
                    features["text"] = row[self.text_column]
                    yield idx, features
                    idx += 1
            else:
                df = pq.read_table(file_, columns=[self.text_column, self.label_column]).to_pandas()
                for id_, (text, labels) in df.iterrows():
                    features = {feature: False for feature in self.classes}
                    features["text"] = text
                    for label in labels:
                        features[label] = True
                    yield idx, features
                    idx += 1
