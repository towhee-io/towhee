# coding : UTF-8

from PIL import Image


def create_image(mode="RGB", size=(200, 200), color=(155, 155, 155)):
    """
    target: create image with specified properties
    method:  use PIL image module
    expected: return image obj
    """
    img = Image.new(mode, size, color)

    return img

