import os
from itertools import chain

import datasets


def fixed_seq_length_of_datasets(
    datasets,
    fixed_seq_length,
    tokenizer,
    load_from_cache_file=False,
):
    block_size = fixed_seq_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        pad_id = tokenizer.pad_id if hasattr(tokenizer, "pad_id") else tokenizer.eos_token_id

        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Padding in front of tokens to align it with the group size.
        if total_length % block_size != 0:
            count_pad_ids = block_size - (total_length % block_size)
            concatenated_examples["input_ids"] = count_pad_ids*[pad_id] + concatenated_examples["input_ids"]
            concatenated_examples["attention_mask"] = count_pad_ids*[0] + concatenated_examples["attention_mask"]

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = datasets.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
        load_from_cache_file=load_from_cache_file,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_datasets


def prepare_wikitext_dataset(
    raw_datasets,
    tokenizer,
    seq_length=512,
    overwrite_cache=False,
):
    column_names = raw_datasets["train"].column_names
    text_column_name = "text"

    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenizer(examples[text_column_name]),
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    lm_datasets = fixed_seq_length_of_datasets(
        tokenized_datasets,
        seq_length,
        tokenizer,
        load_from_cache_file=not overwrite_cache,
    )

    return lm_datasets


def load_dataset(config):
    if "load_dataset" in config:
        dataset = datasets.load_dataset(config["path"], config["name"])
    elif "load_from_disk" in config:
        dataset = datasets.load_from_disk(config["dataset_path"])
    else:
        raise Exception("The configuration is invalid and the dataset cannot be loaded.")
        
    return dataset
