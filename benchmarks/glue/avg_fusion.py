# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import json
import argparse
import logging
import numpy as np
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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_folder",
    type=str,
    default="./results",
    help="results folder",
)
parser.add_argument(
    "--model_folder1",
    type=str,
    default="./model_folder1",
    help="model result folder1",
)
parser.add_argument(
    "--model_folder2",
    type=str,
    default=None,
    help="model result folder2",
)
parser.add_argument(
    "--model_folder3",
    type=str,
    default=None,
    help="model result folder3",
)
parser.add_argument(
    "--model_folder4",
    type=str,
    default=None,
    help="model result folder4",
)
parser.add_argument(
    "--model_folder5",
    type=str,
    default=None,
    help="model result folder5",
)
args, _ = parser.parse_known_args()

model_folder1 = args.model_folder1
model_folder2 = args.model_folder2
model_folder3 = args.model_folder3
model_folder4 = args.model_folder4
model_folder5 = args.model_folder5
results_folder = args.results_folder

if not os.path.exists(results_folder):
    os.makedirs(results_folder, exist_ok=True)

model_folders = [model_folder1]

if model_folder2 is not None:
    model_folders += [model_folder2]

if model_folder3 is not None:
    model_folders += [model_folder3]

if model_folder4 is not None:
    model_folders += [model_folder4]

if model_folder5 is not None:
    model_folders += [model_folder5]


def avg_fusion(*results, task=None):
    results = [result.sort_values("idx") for result in results]
    fusion = results[0]
    if task in ["stsb", "wnli"]:
        fusion["score"] = np.mean([result["score"].values for result in results], axis=0)
    else:
        _results = pd.concat(
            [pd.DataFrame({f"class_score_{i}": result["class_score"]}) for i, result in enumerate(results)],
            axis=1,
        )

        def func(row):
            scores = np.mean(
                [json.loads(v)["scores"] for k, v in row.items() if k.startswith("class_score_")],
                axis=0,
            )
            cls, score = np.argmax(scores), np.max(scores)
            return json.dumps({"class": int(cls), "score": float(score), "scores": scores.tolist()})

        fusion["class_score"] = _results.apply(func, axis=1)
    return fusion


def get_folder_results(folder, sub_file):
    results = []
    for root_dir, dirs, files in os.walk(folder):
        if sub_file in files:
            results.append(os.path.join(root_dir, sub_file))

    return results


for sub_task, sub_file in submission_files.items():
    results = []
    for folder in model_folders:
        results += get_folder_results(folder, sub_file)

    if len(results) == 0:
        logging.info(f"GLUE Task {sub_task} in missing")
        continue

    results = [
        pd.read_csv(
            result_file,
            names=results_headers.get(sub_task),
            sep="\t",
            quoting=3,
        )
        for result_file in results
    ]
    fusion = avg_fusion(*results, task=sub_task)
    fusion.to_csv(
        os.path.join(results_folder, sub_file),
        sep="\t",
        header=None,
        index=False,
        quoting=3,
    )
