import requests
import utils
from PIL import Image

file_name = 'G:/Official_small.jpg'

input_image = Image.open(file_name)
response = requests.post(
    "http://192.168.1.13:8080/api/trans",files={"file":open(file_name,"rb")}, data={"model_name": "celeba"}
    )


output_image = utils.read_image(response.content)

input_image.show()
output_image.show()