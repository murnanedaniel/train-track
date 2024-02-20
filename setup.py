#!/usr/bin/env python
# coding: utf-8

import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


dependencies = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "pyyaml>=5.1",
    "pytorch-lightning",
    "decorator",
    "more_itertools",
    "simple_slurm",
    "memory_profiler",
]

setup(
    name="traintrack",
    version="0.1.6",
    description="A simple helper to run pipelines of PytorchLightning models",
    author="Daniel Murnane",
    install_requires=dependencies,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "traintrack=traintrack.command_line_pipe:main",
            "ttbatch=traintrack.run_pipeline:batch_stage",
        ]
    },
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="Apache License, Version 2.0",
    keywords=[
        "Machine Learning",
        "MLOps",
        "Pytorch",
        "PytorchLightning",
        "Lightning",
        "Pipeline"
    ],
    url="https://github.com/murnanedaniel/train-track",
)
