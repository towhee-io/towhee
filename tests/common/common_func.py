# coding : UTF-8

import os
from towhee.utils.thirdparty.pil_utils import PILImage as Image


def create_image(mode="RGB", size=(200, 200), color=(155, 155, 155)):
    """
    target: create image with specified properties
    method:  use PIL image module
    expected: return image obj
    """
    img = Image.new(mode, size, color)

    return img

def get_all_file_path_from_directory(directory):
    """
    Extract all the file path for one directory
    Parameters:
        directory - directory to be extracted
    Returns:
        a list contains the path for all the files in
        input directory
    """

    filepaths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            filepaths.append(file_path)

    return filepaths

def get_specific_file_path(directory, file_name):
    """
    Extract the path for a specific file name
    Parameters:
        directory - directory to be extracted
        file_name - file name to be extracted
    Returns:
        a list contains the path for the specified file
        in input directory
    """

    descript_paths = []
    filepaths = get_all_file_path_from_directory(directory)
    for filepath in filepaths:
        if file_name in filepath:
            descript_paths.append(filepath)

    return descript_paths
