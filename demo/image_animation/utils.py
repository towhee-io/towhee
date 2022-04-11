from io import BytesIO
from typing_extensions import final
import numpy as np
import towhee
from PIL import Image
import cv2

def read_image(file: BytesIO) -> np.ndarray:
    '''
    Reads the image received in byte format
    and convert into PIL.Image format

    Args:
    - file (BytesIO): Receives a byte stream of file

    Returns:
    - (PIL.Image): A PIL Image object translated from byte stream

    '''
    image = Image.open(BytesIO(file))
    return image

def translate_image(input_img: Image, model: str) -> np.ndarray:
    '''
    Convert the input PIL based on provided model
    using anime gan towhee pipeline.

    Args:
    - input_img (PIL.Image): PIL.Image object
    - model (str): Model to be run on image

    Returns:
    - (PIL.Image): Returns a PIL Image object corresponding to translated image (RGB format)
    '''
    output_img = towhee.ops.img2img_translation.animegan(model_name = model)(input_img)
    out = np.array(output_img)
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    output = Image.fromarray(out_rgb)
    
    return output
