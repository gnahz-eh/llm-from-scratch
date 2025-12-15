"""
Fine-tuning utilities for LLM models.

This module contains utilities for fine-tuning pre-trained language models,
including data downloading, preprocessing, and specialized training routines.
"""

from .data import download_and_unzip_spam_data, create_balanced_dataset, random_split, SpamDataset

__all__ = [
    "download_and_unzip_spam_data",
    "create_balanced_dataset",
    "random_split",
    "SpamDataset"
]

__version__ = "1.0.0"