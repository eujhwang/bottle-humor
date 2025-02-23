from __future__ import annotations
import os
import tqdm
import pandas as pd
from datasets import Dataset
from typing import Any
import torch
import json
import random
from datasets import load_dataset

MEMECAP_GUIDELINE = """### Guidelines for Generating Your Response:  
1. **Focus on Intent:**  
   - Avoid explaining why the meme is funny.  
   - Directly state what the meme poster is attempting to convey (e.g., "The meme poster is...").  

2. **Maintain Clarity:**  
   - Avoid using generic, vague, or ambiguous phrases.  
   - Ensure your sentence is clear and directly connects to the humor, referencing cultural, social, or situational norms being subverted or highlighted by the meme.  

3. **Be Concise and Specific:**  
   - Limit your response to a single, precise sentence.  
   - Focus on capturing the essence of the meme poster's intent without overexplaining or including extraneous details.  
"""

NEWYORKER_GUIDELINE = """### Guidelines for Generating Your Explanation:
1. **Clarity and Specificity:**  
   - Avoid generic or ambiguous phrases.  
   - Provide specific details that connect the roles, contexts, or expectations associated with the elements in the image and its caption.  

2. **Explain the Humor:**  
   - Clearly connect the humor to the caption, image, and any cultural, social, or situational norms being subverted or referenced.  
   - Highlight why the combination of these elements creates an unexpected or amusing contrast.

3. **Prioritize Clarity Over Brevity:**  
   - Justify the humor by explaining all important components clearly and in detail.  
   - Aim to keep your response concise and under 150 words while ensuring no critical details are omitted.  
"""

YESBUT_GUIDELINE = NEWYORKER_GUIDELINE


def load_vflute(data_dir: str, data_type: str):
    print("Load vflute dataset...")

    files = sorted(os.listdir(data_dir))
    train, valid, test = [], [], []
    for file in tqdm.tqdm(files, total=len(files)):
        file_path = os.path.join(data_dir, file)
        if not file.endswith(".parquet"):
            continue

        if data_type in ['train', 'valid', 'test'] and not file.startswith(data_type):
            continue

        df = pd.read_parquet(file_path, engine='pyarrow')
        for _, row in df.iterrows():
            if file.startswith("train"):
                train.append(row.to_dict())
            if file.startswith("valid"):
                valid.append(row.to_dict())
            if file.startswith("test"):
                test.append(row.to_dict())
    return train, valid, test


def load_memecap(data_dir: str, data_type: str):
    """
    keys: item['category'], item['img_captions'], item['meme_captions'], item['title'], item['url'], item['img_fname'], item['metaphors']
    """
    print("Load memecap dataset...")
    # assert data_type == "test"
    with open(os.path.join(data_dir, "memes-trainval.json"), "r+") as f:
        trainval = json.load(f)
        for sample in trainval:
            sample['local_image_path'] = os.path.abspath(os.path.join(data_dir, "images", sample['img_fname']))
            sample['caption'] = sample.pop('title')
            sample['reference'] = sample.pop('meme_captions')

    with open(os.path.join(data_dir, "memes-test.json"), "r+") as f:
        test = json.load(f)
        for sample in test:
            sample['local_image_path'] = os.path.abspath(os.path.join(data_dir, "images", sample['img_fname']))
            sample['caption'] = sample.pop('title')
            sample['reference'] = sample.pop('meme_captions')

    random.shuffle(trainval)
    random.shuffle(test)
    cutoff = int(len(trainval) * 0.95)
    train = trainval[:cutoff]
    valid = trainval[cutoff:]
    return train.copy(), valid.copy(), test.copy()


def load_new_yorker_cartoon(data_dir: str, data_type: str, split: int=0):
    """
    keys: item['image'], item['local_image_path'], item['image_description'], item['image_uncanny_description'], item['entities'],
    item['questions'], item['caption_choices'], item['from_description'], item['label']
    """

    print("Load New Yorker Cartoon dataset...")
    # Available dataset keys: ['explanation', 'explanation_1', 'explanation_2', 'explanation_3', 'explanation_4', 'explanation_from_pixels', 'explanation_from_pixels_1', 'explanation_from_pixels_2', 'explanation_from_pixels_3', 'explanation_from_pixels_4', 'matching', 'matching_1', 'matching_2', 'matching_3', 'matching_4', 'matching_from_pixels', 'matching_from_pixels_1', 'matching_from_pixels_2', 'matching_from_pixels_3', 'matching_from_pixels_4', 'ranking', 'ranking_1', 'ranking_2', 'ranking_3', 'ranking_4', 'ranking_from_pixels', 'ranking_from_pixels_1', 'ranking_from_pixels_2', 'ranking_from_pixels_3', 'ranking_from_pixels_4']
    # matching = load_dataset("jmhessel/newyorker_caption_contest", "matching")
    # ranking = load_dataset("jmhessel/newyorker_caption_contest", "ranking")
    # ranking_from_pixels = load_dataset("jmhessel/newyorker_caption_contest", "ranking_from_pixels")
    # if split == 0:
    data1 = load_dataset("jmhessel/newyorker_caption_contest", "explanation")
    # elif split == 1:
    data2 = load_dataset("jmhessel/newyorker_caption_contest", "explanation_1")
    # elif split == 2:
    data3 = load_dataset("jmhessel/newyorker_caption_contest", "explanation_2")
    # elif split == 3:
    data4 = load_dataset("jmhessel/newyorker_caption_contest", "explanation_3")
    # elif split == 4:
    data5 = load_dataset("jmhessel/newyorker_caption_contest", "explanation_4")
    # else:
    #     raise Exception(f"Invalid split number -- {split}")
    
    valid_items = []
    if data_type == "valid":
        for data in [data1, data2, data3, data4, data5]:
            for item in data['validation']:
                # Encode your PIL Image as a JPEG without writing to disk
                # print(item['contest_number'], item['instance_id'], item['image'])
                image_path_to_save = f"data/newyorker/validation/{item['contest_number']}_{item['instance_id'][:10]}.png"
                if os.path.exists(image_path_to_save):
                    item.update({'local_image_path': os.path.abspath(image_path_to_save)})
                    item['caption'] = item.pop('caption_choices')
                    item['reference'] = item.pop('label')
                    valid_items.append(item.copy())
                    continue
                item['image'].save(image_path_to_save, "PNG")

    test_items = []
    if data_type == "test":
        for data in [data1, data2, data3, data4, data5]:
            for item in data['test']:
                # Encode your PIL Image as a JPEG without writing to disk
                # print(item['contest_number'], item['instance_id'], item['image'])
                image_path_to_save = f"data/newyorker/test/split{split}/{item['contest_number']}_{item['instance_id'][:10]}.png"
                if os.path.exists(image_path_to_save):
                    item.update({'local_image_path': os.path.abspath(image_path_to_save)})
                    item['caption'] = item.pop('caption_choices')
                    item['reference'] = item.pop('label')
                    test_items.append(item.copy())
                    continue
                item['image'].save(image_path_to_save, "PNG")

    print("total number of items:", len(test_items))
    random.shuffle(test_items)
    return [], valid_items.copy(), test_items


def load_yesbut():
    data = load_dataset("bansalaman18/yesbut")
    # print(data)
    test_items = []
    for i, item in enumerate(data['train']):
        # image = item['image']
        image_path_to_save = f"data/yesbut/images/image_{i}.png"
        if not os.path.exists(image_path_to_save):
            item['image'].save(os.path.abspath(image_path_to_save), "PNG")
        item['local_image_path'] = os.path.abspath(image_path_to_save)
        item['reference'] = item.pop('overall_description')
        item['caption'] = 'Yes, But'
        item['difficulty'] = item.pop('difficulty_in_understanding')
        test_items.append(item)
        # print(f"[difficulty]: {difficulty}, [overall_description]: {item['reference']}, [left_image]: {left_image}, [right_image]: {right_image}, [stage]: {stage}")
    random.shuffle(test_items)
    return [], [], test_items



def create_dataset(train: list[Any], valid: list[Any], test: list[Any]):
    train_ds = Dataset.from_list(train)
    valid_ds = Dataset.from_list(valid)
    test_ds = Dataset.from_list(test)
    return train_ds, valid_ds, test_ds


def create_dataloader(train_ds: Dataset, valid_ds: Dataset, test_ds: Dataset, train_batch_size: int,
                      test_batch_size: int):
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=False,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=test_batch_size,
        shuffle=False,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=True,
    )
    return train_dataloader, valid_dataloader, test_dataloader


def initialize_input(caption, image_descriptions, implications, candidate_response, criticism):
    inputs = []
    if caption:
        inputs.append("[Caption]: {caption}".format(caption=caption))
    
    if image_descriptions:
        inputs.append("[Image Descriptions]:\n{image_descriptions}".format(image_descriptions='\n'.join([f"- {x}" for x in image_descriptions])))

    if implications:
        inputs.append("[Implications]:\n{implications}".format(implications='\n'.join([f"- {x}" for x in implications])))
    
    if candidate_response:
        inputs.append("[Candidate Answers]:\n{candidate_response}".format(candidate_response=candidate_response))
    
    if criticism:
        inputs.append("[Feedback for Candidate Answer]:\n{criticism}".format(criticism=criticism))
    
    return "\n\n".join(inputs)


def get_vlm_response_prompt(data_name, caption, image_descriptions=[], implications=[], candidate_response='', add_cot=False, criticism=''):
    input = initialize_input(caption, image_descriptions, implications, candidate_response, criticism)

    if data_name == "memecap":
        guideline = MEMECAP_GUIDELINE
    if data_name == "newyorker":
        guideline = NEWYORKER_GUIDELINE
    if data_name == "yesbut":
        guideline = YESBUT_GUIDELINE

    if image_descriptions or implications or candidate_response or criticism:
        guideline += "\n4. **Use Additional Inputs Effectively:**"""
    
    if image_descriptions:
        guideline += "\n   - **[Image Descriptions]:** Provide a foundation for understanding the visual elements."
    if implications:
        guideline += "\n   - **[Implications]:** Assist in understanding relationships and connections but do not allow them to dominate or significantly alter the central idea."
    if candidate_response:
        guideline += "\n   - **[Candidate Answers]:** Adapt your reasoning by leveraging strengths or improving upon weaknesses in the candidate answers."
    if criticism:
        guideline += "\n   - **[Feedback for Candidate Answer]:** Feedback that points out some weakness in the current candidate responses."
    
    cot_output_instruction = ""
    if add_cot:
        cot_output_instruction = """Begin by analyzing the image and the given context, and explain your reasoning briefly before generating your final response.

Here is an example format of the output:
{{
    "Analysis": "...",
    "Output": "..."  
}}
"""
    
    task_description = ["You are provided with the following inputs:"]
    if data_name == "memecap":
        # Combine all parts into final_sentence
        final_sentence = f"""\n\n### Your Task:\nYour task is to generate **one concise and specific sentence** that clearly conveys what the meme poster is trying to express from their perspective.\n\n{guideline}\n""" + cot_output_instruction
        
        # Build task_description incrementally
        task_description.extend(["- **[Meme]:** A meme posted on Reddit.", "- **[Title]:** The title of a meme as posted by a Reddit user."])
        input = input.replace('[Caption]:', '[Title]:')
        
    elif data_name == "newyorker":
        # Define components based on conditions
        # intro = "Begin by analyzing the image and the given context, and explain your reasoning briefly." if add_cot else ""
    
        # Combine all parts into final_sentence
        final_sentence = f"""\n\n### Your Task:\nGenerate **one concise, specific explanation** that clearly captures why the caption is funny in the context of the image. Your explanation must provide detailed justification and address how the humor arises from the interplay of the caption, image, and associated norms or expectations.\n\n{guideline}\n""" + cot_output_instruction
        
        # Build task_description incrementally
        # task_description.extend(["You are given a New Yorker cartoon image along with its [Caption], written by a human"])
        task_description.extend(["- **[Image]:** A New Yorker cartoon image.", "- **[Caption]:** A caption written by a human to accompany the image."])
    
    elif data_name == "yesbut":    
        # Combine all parts into final_sentence
        final_sentence = f"""\n\n### Your Task:\nGenerate **one concise, specific explanation** that clearly captures why the caption is funny in the context of the image. Your explanation must provide detailed justification and address how the humor arises from the interplay of the caption, image, and associated norms or expectations.\n\n{guideline}\n""" + cot_output_instruction
        
        # Build task_description incrementally
        task_description.extend(["- **[Image]:** Yes-But image.", "- **[Caption]:** A caption accompanying the image."])
                
    else:
        raise Exception(f"Invalid data name -- {data_name}")
    
    # Incrementally add to task description based on available items
    if image_descriptions:
        task_description.append("- **[Image Descriptions]:** Literal descriptions of the visual elements in the image.")
    if implications:
        task_description.append("- **[Implications]:** Possible connections or relationships between objects, concepts, or the caption and the image.")
    if candidate_response:
        task_description.append("- **[Candidate Answers]:** Example answers generated in a previous step to provide guidance and context.")
    if criticism:
        task_description.append("- **[Feedback for Candidate Answer]:** Feedback that points out some weakness in the current candidate responses.")

    # Join task_description parts with commas and add "and" before the last item
    if len(task_description) > 1:
        task_description_text = "\n".join(task_description) #', '.join(task_description[:-1]) + ", and " + task_description[-1]
    else:
        task_description_text = task_description[0]

    # Combine task_description and final_sentence
    task_description_text += final_sentence
    
    if data_name == "memecap":
        final_prompt = '\n'.join([task_description_text, f"Now, proceed to generate your response based on the provided inputs.\n\n### Inputs:\n{input}\n\n[Output]:"])
    else:
        final_prompt = '\n'.join([task_description_text, f"Now, proceed to generate your response based on the provided inputs.\n\n### Inputs:\n{input}\n\n[Output]:"])
    
    # print("[final_prompt]:\n", final_prompt)
    # if candidate_response:
    #     assert False
    return final_prompt #'\n\n'.join([task_description_text, input, "Output:\n"])



def get_cross_entropy_response_prompt(data_name, caption, image_descriptions, implications=[]): # used only when implications and candidate answers are avilable
    if isinstance(implications, str):
        implications = [implications]
    if data_name == "newyorker":
        # task_description = "Your task is to generate meaningful implications that reveal the relationships between concepts in the image helping to understand why the caption is funny for an image."
        final_task_description = "Your task is to generate a specific explanation that describes why the caption is funny for an image."
        guideline = """
Here are some guidelines when generating your response:
{NEWYORKER_GUIDELINE}
"""
    elif data_name == "memecap":
        # task_description = "Your task is to generate implications that reveal the relationships between concepts in the image helping to understand what the meme poster is trying to convey."
        final_task_description = "Your task is to generate a concise, specific sentence that explains what the meme poster is trying to convey."
        guideline = f"""
Here are some guidelines when generating your response:
{MEMECAP_GUIDELINE}
"""
    elif data_name == "yesbut":
        # task_description = "Your task is to generate implications that reveal the relationships between concepts in the image helping to understand why the image is funny or satirical."
        final_task_description = "Your task is to generate a specific explanation that describes why the image is funny or satirical."
        guideline = f"""
Here are some guidelines when generating your response:
{YESBUT_GUIDELINE}
"""
    else:
        raise Exception(f"Invalid data name -- {data_name}")

    if implications:
        input_prompt = """You are given with a [Caption], [Image Descriptions], and [Implications] that describe possible connections across the objects or concepts in the image descriptions and the caption.
{final_task_description}
{guideline}
[Caption]: {caption} 
[Image Descriptions]:
{image_descriptions}

[Implications]:
{implications}

Proceed to generate your response.""".format(final_task_description=final_task_description, guideline=guideline, caption=caption, image_descriptions='\n'.join(image_descriptions), implications='\n'.join(implications))
    else:
        final_task_description = final_task_description.replace("Your task is ", "")
        input_prompt = """You are given with a [Caption], and [Image Descriptions] that literally describe the image.
Your ultimate goal is {final_task_description} However, before arriving at this explanation, you must first uncover and articulate the implicit meanings or connections between the objects, characters, situations, or expressions in the caption and image descriptions. Generate insightful implications that illuminate these implicit connections to enhance understanding of the humor or deeper meaning in the meme.

[Caption]: {caption} 
[Image Descriptions]:
{image_descriptions}

Proceed to generate your response.""".format(final_task_description=final_task_description, guideline=guideline, caption=caption, image_descriptions='\n'.join(image_descriptions))

    system_prompt = "You are a helpful AI assistant. You generate answers that are specific, contextually grounded, and insightful. Your goal is to maximize understanding of the humor or connections in the image and caption."
#     final_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
# {input_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """
    final_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|><|im_start|>user
{input_prompt}<|im_end|><|im_start|>assistant
"""

    # print("[final_prompt]:\n", final_prompt)
    return final_prompt