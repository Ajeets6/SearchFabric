"""
Data Module
"""

from data.file_processor import (
    FileProcessor,
    SUPPORTED_TEXT,
    SUPPORTED_IMAGE,
    SUPPORTED_PDF,
    MAX_FILE_SIZE_MB
)

__all__ = [
    'FileProcessor',
    'SUPPORTED_TEXT',
    'SUPPORTED_IMAGE',
    'SUPPORTED_PDF',
    'MAX_FILE_SIZE_MB'
]
