# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from collections import OrderedDict
from lazydocs import generate_docs

generate_docs(
    ["../"],
    output_path="./references",
    src_root_path="../",
    src_base_url="https://github.com/fuliucansheng/unitorch/blob/master/",
    ignored_modules=[
        "unitorch.cli",
        "unitorch.microsoft",
        "unitorch.scripts",
        "unitorch.services",
        "unitorch.modules.replace",
        "unitorch.modules.prefix_model",
        "unitorch.models.detectron2.backbone",
        "unitorch.models.detectron2.meta_arch",
        "unitorch.ops",
        "unitorch.optim",
        "unitorch.task",
        "unitorch.writer",
        "unitorch_cli",
    ],
)


def get_folder_files(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".md") and f.lower() != "readme.md"]
    files = [os.path.normpath(f) for f in files]
    return sorted(files, key=lambda x: x[:-3])


def get_markdown_docs(files, space=0):
    docs = []
    for f in files:
        name = os.path.basename(f)[:-3]
        link = f"/{f}"
        name, link = name.replace("_", "\_"), link.replace("_", "\_")
        docs += [f"{' ' * space}* [**{name}**]({link})"]
    return docs


sidebar_docs = []

## Quick Start
quick_start = "* [**Quick Start**](/quick-start.md)"

sidebar_docs += [quick_start]

## Installation
installation = "* [**Installation**](/install.md)"

sidebar_docs += [installation]

## Overview
introduction = "* **Introduction**"
overview = "  * [**overview**](/overview.md)"
optimized_techniques = "  * [**optimized techniques**](/optimized_techniques.md)"
configuration = "  * [**configuration**](/configuration.md)"

sidebar_docs += [introduction, overview, optimized_techniques, configuration]

## Tutorials
tutorials = "* [**Tutorials**](/tutorials/readme.md)"
tutorials_files = get_folder_files("./tutorials")

sidebar_docs += [tutorials] + get_markdown_docs(tutorials_files, space=2)

## References
references = "* [**References**](/references/readme.md)"
references_files = get_folder_files("./references")


def generate_reference_readme(files):
    docs = get_markdown_docs(files, space=0)

    title = "## unitorch documents"
    description = "unitorch provides efficient implementation of popular unified NLU / NLG / CV / MM / RL models with PyTorch. It automatically optimizes training / inference speed based on pupular deep learning toolkits (transformers, fairseq, fastseq, etc) without accuracy loss."
    notes = "> Notes"

    contents = [title, description, notes] + docs
    with open("./references/readme.md", "w") as f:
        f.write("\n".join(contents))


def process_reference_file(doc_files):
    replace_dict = OrderedDict(
        {
            ": # Copyright (c) FULIUCANSHENG.": "",
            "# ": "### ",
            "**Global Variables**": "### **Global Variables**",
        }
    )

    def process_contents(contents):
        new_contents = []
        for content in contents:
            if "lazydocs" in content.lower():
                continue

            if "licensed under the mit license." in content.lower():
                continue

            for k, v in replace_dict.items():
                content = content.replace(k, v)

            new_contents.append(content)
        return new_contents

    doc_file = doc_files[0]
    doc_contents = open(doc_file, "r").read().split("\n")
    file_pointer = open(doc_file, "w")

    write_contents = process_contents(doc_contents)
    for more_doc_file in doc_files[1:]:
        doc_contents = open(more_doc_file, "r").read().split("\n")
        write_contents += process_contents(doc_contents)
        os.remove(more_doc_file)
    file_pointer.write("\n".join(write_contents))


def is_single(path):
    mfile = os.path.basename(path)
    if mfile in [
        "unitorch.md",
        "unitorch.cli.md",
        "unitorch.cli.models.md",
        "unitorch.models.md",
    ]:
        return True
    if mfile.startswith("unitorch.cli.models"):
        return mfile.count(".") == 4

    if mfile.startswith("unitorch.cli"):
        return mfile.count(".") == 3

    if mfile.startswith("unitorch.models"):
        return mfile.count(".") == 3

    if mfile.startswith("unitorch_cli"):
        return mfile.count(".") == 1

    if mfile.startswith("unitorch"):
        return mfile.count(".") == 2

    return mfile.count(".") == 1


index = 0
while index < len(references_files):
    process_files = [references_files[index]]
    index += 1
    while index < len(references_files) and not is_single(references_files[index]):
        process_files.append(references_files[index])
        index += 1
    print(process_files)
    process_reference_file(process_files)

new_references_files = get_folder_files("./references")
generate_reference_readme(new_references_files)

sidebar_docs += [references] + get_markdown_docs(new_references_files, space=2)

## Benchmarks
benchmarks = "* [**Benchmarks**](/benchmarks/readme.md)"
benchmarks_files = get_folder_files("./benchmarks")

sidebar_docs += [benchmarks] + get_markdown_docs(benchmarks_files, space=2)

with open("./_sidebar.md", "w") as f:
    f.write("\n".join(sidebar_docs))
