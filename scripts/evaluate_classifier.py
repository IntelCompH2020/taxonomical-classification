import torch
import datasets
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from sklearn.metrics import classification_report
from transformers import (
    HfArgumentParser, 
    pipeline, 
    AutoConfig, 
    AutoTokenizer
    AutoModelForSequenceClassification, 
)


@dataclass
class DataArguments:
    data_loading_script: str = field(
        default=None,
        metadata={"help": "Path to the data loading script."}
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
    output_file: str = field(
        default=None,
        metadata={"help": "Path to the output file to write the classification report."}
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer or not."},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Load data
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

    # Load model and tokenizer
    print(">>> loading model...")
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        model_max_length = 512,
    )
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512,'return_tensors':'pt'}
    print(">>> model loaded :)")

    # Define the  pipeline
    print(">>> loading classifier...")
    if torch.cuda.is_available():
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
        print(">>> classifier loaded to GPU :)")
    else:
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
        print(">>> classifier loaded to CPU :)")

    print(">>> Evaluating on test data...")
    gs_labels = []
    pred_labels = []
    for example in tqdm(raw_datasets["test"]):
        candidate_labels = [fos for fos in example.keys() if fos != "text"]
        gs_label = [1 if example[label] else 0 for label in candidate_labels]
        gs_labels.append(gs_label)
        prediction = classifier(example["text"], return_all_scores=True, **tokenizer_kwargs)
        preds = [pred["label"] for pred in prediction]
        pred_label = [1 if label in preds else 0 for label in candidate_labels]
        pred_labels.append(pred_label)

    with open(data_args.output_file, 'w') as f:
        f.write(classification_report(gs_labels, pred_labels, digits=4))
    print(">>> test evaluation completed :)")


if __name__ == "__main__":
    main()