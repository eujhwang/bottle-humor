import json
import ast
import pandas as pd
from PIL import Image
from io import BytesIO

import base64
from mimetypes import guess_type
import re
from nltk.tokenize import sent_tokenize


def convert_image_bytes_to_image(bytes: str, savepath: str):
    image = Image.open(BytesIO(bytes))  # Load image from BytesIO
    image.save(savepath)  # Save image


# Function to encode a local image into data URL
def local_image_to_data_url(image_path, format=True):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    if format:
        return f"data:{mime_type};base64,{base64_encoded_data}", None
    return base64_encoded_data, mime_type


def load_single_parquet_file(files):
    if isinstance(files, list):
        items = []
        for file in files:
            df = pd.read_parquet(file, engine='pyarrow')
            for index, row in df.iterrows():
                items.append(row.to_dict())
        return items
    elif isinstance(files, str):
        df = pd.read_parquet(files, engine='pyarrow')
        items = []
        for index, row in df.iterrows():
            items.append(row.to_dict())
        return items
    else:
        raise Exception(f"Invalid type of files: {type(files)}")


def parse_json_output(json_str):
    json_str = json_str.strip()
    if json_str.startswith("```json"):
        json_str = json_str.replace('`', '').replace('json', '').strip()
        if json_str.lower() == 'none' or json_str.lower() == 'null':
            return ''
    elif json_str.startswith("```text"):
        json_str = json_str.replace('```text', '').replace('`', '').strip()
        return json_str
    elif json_str.startswith("```"):
        json_str = json_str.replace('`', '').strip()
        return json_str
    else:
        return json_str

    try:
        data = json.loads(json_str)
        return data
        # label = data['label']
        # explanation = data['explanation']
        # return label, explanation
    except json.JSONDecodeError as e:
        if '"explanation":' in json_str and '"short_sentence":' in json_str:
            explanation = json_str[json_str.find('"explanation":'):json_str.find('"short_sentence":')]
            explanation = explanation.replace('"explanation":', '').replace('"', '').strip()
            short_sentence = json_str[json_str.find('"short_sentence":'):json_str.rfind('"')]
            short_sentence = short_sentence.replace('"short_sentence":', '').replace('"', '').strip()
            return {
                "explanation": explanation,
                "short_sentence": short_sentence,
            }
        print(f"Error decoding JSON: {e}. Original json str: {json_str}")
        return ''
    except KeyError as e:
        print(f"KeyError: {e} not found in JSON. Original json str: {json_str}")
        return ''


def parse_python_output(str):
    if "```python" in str:
        str = str[str.find("```python"):str.rfind("```")]

    try:
        str = str.replace('```python', '').replace('`', '').strip()
        data = ast.literal_eval(str)
        return data
    except Exception as e:
        print(f"Error: {e}")
        return str


def read_prompt(file_path):
    with open(file_path, "r") as f:
        return f.read()
    

def sentence_tokenize(response):
    # below handles the case of missing dot between double quotes and a new sentence starting with uppercase letter (e.g.  A caption below the .. "Corporate ... this song" A woman ...)
    response = re.sub(r'(\.\")(\s*)([A-Z])', r'.".\2\3', response)
    response = response.split("\n")
    _response = []
    for x in response:
        tokenized_sentences = sent_tokenize(x)
        if len(tokenized_sentences) == 1:
            # print("check1:", tokenized_sentences)
            _response.append(x)
        else:
            _sent = ""
            for _, tok_sent in enumerate(tokenized_sentences):
                if _sent:
                    if len(tok_sent.split(" ")) <= 3 or (_sent.count('"') % 2 == 1) or not _sent.endswith("."): # if _sent exists, but length of split is less than 3 or number of double quotes is odd, then append current tok_sent to the previous sequence.
                        _sent += (" " + tok_sent)
                        _response.pop(-1)
                        # print("check3-2:", _sent)
                        _response.append(_sent)
                    else: # if _sent exists, but tok_sent is also valid -> new sentence appeared
                        _sent = tok_sent
                        # print("check3-3:", _sent)
                        _response.append(_sent)
                else: # if _sent doesnt exist, _sent is tok_sent
                    _sent += tok_sent
                    # print("check3-1:", _sent)
                    _response.append(_sent)
    return _response

def parse_cot_response(response):
    response = response.strip() #.replace("Reasoning:", "").replace("Final Answer:", "").replace("Explanation:", "").replace("*", "").strip()
    if not response:
        return "", ""
    reasoning, final_response = response, response
    if '"Reasoning"' in response:
        if '"Explanation"' in response:
            final_response = response[response.index('"Explanation"'):].replace('"Explanation":', '').replace("}", "").strip()
        elif '"Output"' in response:
            final_response = response[response.index('"Output"'):].replace('"Output":', '').replace("}", "").strip()
    else:
        final_response = response
    
    final_response.strip()
    if final_response.strip().startswith('"') or final_response.strip().startswith("'"):
        final_response = final_response.strip()[1:]
    if final_response.strip().endswith("'") or final_response.strip().endswith('"'):
        final_response = final_response.strip()[:-1]
        
    return reasoning, final_response


def reduce_image_size(image_path, output_path, reduction_percentage=10):
    """
    Reduces an image's dimensions by a given percentage without losing quality.

    Args:
        image_path (str): Path to the original image.
        output_path (str): Path to save the resized image.
        reduction_percentage (int): Percentage to reduce dimensions (e.g., 10 for 10%).
    """
    try:
        # Open the original image
        with Image.open(image_path) as img:
            # Calculate new dimensions
            width, height = img.size
            new_width = int(width * (1 - reduction_percentage / 100))
            new_height = int(height * (1 - reduction_percentage / 100))
            
            # Resize with LANCZOS filter for high-quality downscaling
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            # Save the image in its original format
            img_format = img.format  # Preserve original file format
            resized_img.save(output_path, format=img_format, optimize=True, quality=100)  # Keep quality high
            print(f"Image reduced by {reduction_percentage}%. Original: {width}x{height}, New: {new_width}x{new_height}.")
    except Exception as e:
        print(f"Error resizing image: {e}")

        
def resave_image(image_path, output_path, format="PNG"):
    """
    Resaves an image to a clean, standard format.

    Args:
        image_path (str): Path to the original image.
        output_path (str): Path to save the corrected image.
        format (str): Desired format (e.g., "JPEG", "PNG").
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Convert to RGB to ensure compatibility (if required)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save the image in the desired format, stripping metadata
            img.save(output_path, format=format, optimize=True)
        print(f"Image successfully resaved as {output_path}.")
    except Exception as e:
        print(f"Error resaving image: {e}")
