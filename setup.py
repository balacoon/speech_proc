# -*- coding: utf-8 -*-
# Copyright 2022 Balacoon

from setuptools import setup, find_packages


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="speech_proc",
    version="0.0.1",
    author="Clement Ruhm",
    author_email="clement@balacoon.com",
    description="Speech processing and feature extraction",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://balacoon.com/",
    python_requires=">=3.10",
    # declare your packages
    packages=find_packages(where="src", exclude=("tests",)),
    package_dir={"": "src"},
    # declare your scripts
    entry_points="""\
     [console_scripts]
     phoneme-recognizer = speech_proc.phoneme_recognizer:main
     nano-acoustic-tokens = speech_proc.nano_acoustic_tokens:main
     qwen3-acoustic-tokens = speech_proc.qwen3_acoustic_tokens:main
     fundamental-frequency = speech_proc.pitch:main
     speaker-embedding = speech_proc.spkr_emb:main
    """
)
