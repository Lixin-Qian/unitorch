# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import platform
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

UNITORCH_VERSION = "0.0.0.1"


def get_unitorch_version():
    return UNITORCH_VERSION


extras = {}

extras["deepspeed"] = ["deepspeed==0.5.8"]
extras["onnxruntime"] = [
    "onnxruntime==0.5.8",
    "torch-ort",
]
extras["detection"] = [
    "torch==1.7.1",
    "torchvision==0.8.2",
    "opencv-python==4.4.0.44",
    "detectron2 @ git+https://github.com/facebookresearch/detectron2@v0.6",
]

ngram_cuda_extension = CUDAExtension(
    "unitorch.clib.ngram_repeat_block_cuda",
    [
        "unitorch/clib/cuda/ngram_repeat_block_cuda.cpp",
        "unitorch/clib/cuda/ngram_repeat_block_cuda_kernel.cu",
    ],
)

install_requires = open("requirements.txt", "r").read().split("\n")


def do_setup(package_data, package_extensions):
    setup(
        name="unitorch",
        version=get_unitorch_version(),
        author="fuliucansheng",
        author_email="fuliucansheng@gmail.com",
        description="Unified Toolkit based PyTorch",
        long_description=open("readme.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords="PyTorch",
        license="MIT",
        url="https://github.com/fuliucansheng/unitorch",
        packages=find_packages(where=".", exclude=["tests", "__pycache__"]),
        include_package_data=True,
        package_data=package_data,
        setup_requires=[
            "cython",
            "numpy",
            "setuptools>=18.0",
        ],
        install_requires=install_requires,
        extras_require=extras,
        python_requires=">=3.6.0",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        ext_modules=package_extensions,
        entry_points={
            "console_scripts": [
                "unitorch-train = unitorch_cli.train:cli_main",
                "unitorch-infer = unitorch_cli.infer:cli_main",
                "unitorch-script= unitorch_cli.script:cli_main",
                "unitorch-service = unitorch_cli.service:cli_main",
            ],
        },
        cmdclass={"build_ext": BuildExtension},
    )


def get_files(path, relative_to="unitorch"):
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        root = os.path.relpath(root, relative_to)
        for file in files:
            if file.endswith(".pyc"):
                continue
            all_files.append(os.path.join(root, file))
    return all_files


if __name__ == "__main__":
    extensions = []
    link_folders = ["microsoft", "examples", "benchmarks", "scripts", "services"]
    try:
        # symlink folders into unitorch package so package_data accepts them
        for folder in link_folders:
            unitorch_folder = os.path.join("unitorch", folder)
            if not os.path.exists(unitorch_folder):
                os.symlink(os.path.join("..", folder), unitorch_folder)

        package_data = {
            "unitorch": sum(
                [
                    get_files(os.path.join("unitorch", folder))
                    for folder in link_folders
                ],
                [],
            )
        }

        if "--enable_ngram_cuda_extension" in sys.argv[1:]:
            extensions.append(ngram_cuda_extension)
            sys.argv.remove("--enable_ngram_cuda_extension")

        do_setup(package_data, extensions)
    finally:
        for folder in link_folders:
            unitorch_folder = os.path.join("unitorch", folder)
            if "build_ext" not in sys.argv[1:] and os.path.islink(unitorch_folder):
                os.unlink(unitorch_folder)
