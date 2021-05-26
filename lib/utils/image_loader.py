"""Load image pairs for each task.
"""
import logging
import os

def load_images(text_file: str):
    """Load images from text file.
    Args:
        * text_file: Text file containing the list of image IDs.
    Returns:
        * images: The list of all images.
    """
    with open(text_file, "r") as handle:
        images = [x.rstrip() for x in handle.readlines()]
    logging.info(f"Found {len(images)}")
    return images


def load_image_pairs(text_file: str):
    """Load image pairs from text file.
    Args:
        * text_file: Text file containing the list of image IDs.
    Returns:
        * images: The list of all unique images.
        * pairs: The list of image pairs.
    """
    with open(text_file, "r") as handle:
        images = [x.rstrip() for x in handle.readlines()]
        pairs = [x.split(",") for x in images]
        images = list(set([item for sublist in pairs for item in sublist]))

    logging.info(f"Found {len(images)} unique images, and {len(pairs)} image pairs.")
    return images, pairs
