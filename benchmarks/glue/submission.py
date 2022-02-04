# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import json
import argparse
import logging
import pandas as pd

submission_files = {
    "cola": "CoLA.tsv",
    "sst2": "SST-2.tsv",
    "mrpc": "MRPC.tsv",
    "stsb": "STS-B.tsv",
    "mnli": "MNLI-m.tsv",
    "mnli_mismatched": "MNLI-mm.tsv",
    "qnli": "QNLI.tsv",
    "qqp": "QQP.tsv",
    "rte": "RTE.tsv",
    "wnli": "WNLI.tsv",
    "ax": "AX.tsv",
}

results_headers = {
    "cola": ["sentence", "idx", "class_score"],
    "sst2": ["sentence", "idx", "class_score"],
    "mrpc": ["sentence1", "sentence2", "idx", "class_score"],
    "stsb": ["sentence1", "sentence2", "idx", "score"],
    "mnli": ["premise", "hypothesis", "idx", "class_score"],
    "mnli_mismatched": ["premise", "hypothesis", "idx", "class_score"],
    "qnli": ["question", "sentence", "idx", "class_score"],
    "qqp": ["question1", "question2", "idx", "class_score"],
    "rte": ["sentence1", "sentence2", "idx", "class_score"],
    "wnli": ["sentence1", "sentence2", "idx", "score"],
    "ax": ["premise", "hypothesis", "idx", "class_score"],
}

label_classes = {
    "cola": ["0", "1"],
    "sst2": ["0", "1"],
    "mrpc": ["0", "1"],
    "stsb": None,
    "mnli": ["entailment", "neutral", "contradiction"],
    "mnli_mismatched": ["entailment", "neutral", "contradiction"],
    "qnli": ["entailment", "not_entailment"],
    "qqp": ["0", "1"],
    "rte": ["entailment", "not_entailment"],
    "wnli": ["0", "1"],
    "ax": ["entailment", "neutral", "contradiction"],
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_folder",
    type=str,
    default="./results",
    help="results folder",
)
parser.add_argument(
    "--submission_folder",
    type=str,
    default="./submission",
    help="submission folder",
)
args, _ = parser.parse_known_args()

submission_folder = args.submission_folder

results_folder = args.results_folder

if not os.path.exists(submission_folder):
    os.makedirs(submission_folder, exist_ok=True)


def get_submission_file(result, task):
    if task == "stsb":
        result["label"] = result["score"].map(lambda x: min(max(round(x), 0), 5))
    elif task == "wnli":
        labels = label_classes[task]
        gate = result["score"].quantile(0.75)
        result["label"] = result["score"].map(lambda x: labels[x >= gate])
    else:
        labels = label_classes[task]
        result["label"] = result["class_score"].map(
            lambda x: labels[json.loads(x)["class"]]
        )
    result = result[["idx", "label"]]
    return result


for sub_task, sub_file in submission_files.items():
    result_file = os.path.join(results_folder, sub_file)
    if not os.path.exists(result_file):
        logging.info(f"GLUE Task {sub_task} in {results_folder} missing")
        continue

    results = pd.read_csv(
        result_file,
        names=results_headers.get(sub_task),
        sep="\t",
        quoting=3,
    )
    submission = get_submission_file(results, sub_task)
    submission.columns = ["index", "prediction"]
    submission.to_csv(
        os.path.join(submission_folder, sub_file),
        sep="\t",
        index=False,
        quoting=3,
    )
