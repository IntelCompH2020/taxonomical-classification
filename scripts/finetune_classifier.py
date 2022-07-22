#!/usr/bin/env python
# coding=utf-8

import os
import sys
import random
import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

import torch
import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    EvalPrediction,
    default_data_collator,
    DataCollatorWithPadding,
    PretrainedConfig,
    set_seed,
)

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """ Arguments pertaining to the data that will be fed to the model. """

    taxonomy: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the taxonomy to train on."}
    )
    data_loading_script: str = field(
        default="data_loader.py",
        metadata={"help": "The path to the data loading script."}
    )
    text_column: str = field(
        default=None,
        metadata={"help": "Name of the column with the text data."}
    )
    label_column: str = field(
        default=None,
        metadata={"help": "Name of the column with the label name."}
    )
    train_set_path: str = field(
        default=None,
        metadata={"help": "Path to the training dataset."}
    )
    dev_set_path: str = field(
        default=None,
        metadata={"help": "Path to the development dataset."}
    )
    test_set_path: str = field(
        default=None,
        metadata={"help": "Path to the test dataset."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of training examples for debugging purposes or quicker training."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of evaluation examples for debugging purposes or quicker training."}
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of prediction examples for debugging purposes or quicker training."}
    )

    def __post_init__(self):
        if self.taxonomy is not None:
            self.taxonomy = self.taxonomy.lower()


@dataclass
class ModelArguments:
    """ Arguments pertaining to the model/config/tokenizer to be fine-tuned. """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    problem_type: Optional[str] = field(
        default="multi_label_classification", 
        metadata={"help": "Can be either single_label_classification or multi_label_classification."}
    )
    cache_dir: Optional[str] = field(
        default=".cache",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer or not."},
    )


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set verbosity levels
    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log a small summary on each process
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detect the last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model
    set_seed(training_args.seed)

    # Load the data
    print(">>> loading data...")
    raw_datasets = datasets.load_dataset(
        path               = data_args.data_loading_script, 
        path_to_train_data = data_args.train_set_path, 
        path_to_dev_data   = data_args.dev_set_path, 
        path_to_test_data  = data_args.test_set_path, 
        text_column        = data_args.text_column,
        label_column       = data_args.label_column,
        cache_dir          = model_args.cache_dir,
    )
    print(">>> data loaded :)")
    print(raw_datasets)

    # List of labels, sorted for determinism
    label_list = [label for label in raw_datasets['train'].features.keys() if label not in ["text"]]
    label_list.sort()

    num_labels = len(label_list)
    id_to_label = {idx:label for idx, label in enumerate(label_list)}
    label_to_id = {label:idx for idx, label in enumerate(label_list)}

    # Load pretrained model and tokenizer
    print(">>> loading model and tokenizer...")
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label=id_to_label,
        finetuning_task="text-classification",
        problem_type=model_args.problem_type,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    print(">>> model and tokenizer loaded :)")

    # Specify sentence keys (only one sentence is required for text classification tasks)
    sentence1_key, sentence2_key = "text", None

    # Set padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max_seq_len in each batch
        padding = False
    print(">>> padding:", padding)

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(f"max_seq_length is larger than the max length for the model. Using max_seq_length={tokenizer.model_max_length}.")
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    print(">>> max_seq_len:", max_seq_length)

    # Function to preprocess the data
    def preprocess_function(examples):
        # take a batch of texts
        text = examples["text"]
        # encode them
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in label_list}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), num_labels))
        # fill numpy array
        for idx, label in enumerate(label_list):
            labels_matrix[:, idx] = labels_batch[label]
        # add to dictionary as list
        encoding["labels"] = labels_matrix.tolist()
        return encoding

    print(">>> maping datasets...")
    # with training_args.main_process_first(desc="dataset map pre-processing"):
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        remove_columns=raw_datasets['train'].column_names,
        desc="Running tokenizer on dataset",
    )
    print(">>> datasets mapped :)")

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Function to calculate F1 score in multilabel setup
    def multi_label_metrics(predictions, true_labels, threshold=0.5):
        # Apply sigmoid on predictions, which have shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # Use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # Compute metrics
        f1_micro_average = f1_score(y_true=true_labels, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(true_labels, y_pred, average = 'micro')
        accuracy = accuracy_score(true_labels, y_pred)
        metrics = {'f1': f1_micro_average, 'roc_auc': roc_auc, 'accuracy': accuracy}
        return metrics

    # Custom compute_metrics function
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(predictions=preds, true_labels=p.label_ids)
        return result

    # Select data collator, defaults to DataCollatorWithPadding when the tokenizer is passed to Trainer
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    print(">>> data_collator selected :)")

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = (data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset))
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Removing the `label` columns because it contains -1s
        predict_dataset = predict_dataset.remove_columns("labels")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{data_args.taxonomy}.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()
