import os
import base64
from PIL import Image
from io import BytesIO

# def base64_decode(encoded_string):
#     try:
#         # Convert the encoded string to bytes
#         encoded_bytes = encoded_string.encode('utf-8')

#         # Decode the base64-encoded bytes
#         decoded_bytes = base64.b64decode(encoded_bytes)
#         from io import BytesIO
#         image_stream = BytesIO(decoded_bytes)

#         # Open the image using PIL
#         image = Image.open(image_stream)


#         # Convert the decoded bytes to a string
#         decoded_string = decoded_bytes.decode('utf-8')

#         return decoded_string
#     except Exception as e:
#         print("An error occurred during base64 decoding:", str(e))
#         return None


data=os.listdir("/mnt/petrelfs/liushuai1/SG_VLM/OctoGibson_data/all_data_images")
import json
with open("/mnt/petrelfs/liushuai1/SG_VLM/OctoGibson_data/OctoGibson_images.json","r")as f:
    og=json.load(f)


