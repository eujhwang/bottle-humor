from abc import ABC, abstractmethod
from collections import OrderedDict
import json
import os
from datetime import datetime
import pytz
from sklearn.metrics.pairwise import cosine_similarity
from utils.model_utils import get_sentence_transformer_embedding, calculate_cosine_similarity, calculate_batched_cross_entropy
from utils.data_utils import load_memecap, load_new_yorker_cartoon, load_yesbut, get_vlm_response_prompt, get_cross_entropy_response_prompt
from utils.utils import read_prompt, sentence_tokenize, parse_cot_response
import torch
from sentence_transformers import util as st_util
import random
import re
import tqdm
import numpy as np
from fuzzywuzzy.process import dedupe
from fuzzywuzzy import fuzz
import logging
logging.getLogger().setLevel(logging.ERROR)

class Base(ABC):
    def __init__(
        self, model_name, data_dir, data_type, seed, add_cot=True, add_image_descriptions=False, add_implications=False, add_candidate_response=False, overwrite=False, temperature=0.0, weight=0.6, num_hops=2, 
        out_dir_name='', split=0, self_improve=False, advanced_model=None, selection_method=""
    ):
        self.model_name = model_name
        self.data_dir = data_dir
        self.data_type = data_type
        self.data_name = os.path.abspath(self.data_dir).split("/")[-1]
        self.seed = seed
        
        self.add_cot = add_cot
        self.add_image_descriptions = add_image_descriptions
        self.add_implications = add_implications
        self.add_candidate_response = add_candidate_response
        self.overwrite = overwrite
        self.split = split
        self.selection_method = selection_method
        
        self.temperature = temperature # should temperature be constant for every generation? yes
        self.weight = weight
        self.num_hops = num_hops
        self.cosine_threshold = 0.85
        self.advanced_model = None
        self.self_improve = self_improve
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # initialize directory to save model output generations
        exp_type = ""
        if self.add_cot: 
            exp_type += "-cot"
        if self.add_image_descriptions:
            exp_type += "-desc"
        if self.add_implications:
            exp_type += "-imp"
        if self.add_candidate_response:
            exp_type += "-cand"
        exp_type = exp_type.strip("-") if exp_type else "none"
        
        if not out_dir_name:
            self.out_dir_name = datetime.now(pytz.timezone("US/Pacific")).strftime("%Y%m%d")
        else:
            self.out_dir_name = out_dir_name
        
        if self.self_improve:
            self.file_save_dir = os.path.join(self.data_dir, "model-output", self.name(), "self-improve")
        else:
            self.file_save_dir = os.path.join(self.data_dir, "model-output", self.name(), exp_type, self.out_dir_name)
        
        os.makedirs(self.file_save_dir, exist_ok=True)
        self.file_save_path = os.path.abspath(os.path.join(self.file_save_dir, f"model_output_{data_type}_t{temperature}_w{weight}_h{num_hops}_s{split}_sd{seed}{self.selection_method}.json"))
        print("Output generations will be saved to here: {}".format(self.file_save_path))
            
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def name(self):
        pass

    # generation
    @abstractmethod
    def send_request(self, text, image_path=None, max_new_tokens=512, temperature=0.0):
        pass

    def load_dataset(self):
        print("[self.data_name]:", self.data_name)
        if "memecap" == self.data_name:
            self.train_data, self.validation_data, self.test_data = load_memecap(data_dir=self.data_dir, data_type=self.data_type)
        elif "newyorker" == self.data_name:
            self.train_data, self.validation_data, self.test_data = load_new_yorker_cartoon(
                data_dir=self.data_dir, data_type=self.data_type, split=self.split
            )
        elif "yesbut" == self.data_name:
            self.train_data, self.validation_data, self.test_data = load_yesbut()
        else:
            raise Exception(f"Invalid data name -- {self.data_name}")
        print("train: {}, valid: {}, test: {}".format(len(self.train_data), len(self.validation_data), len(self.test_data)))
    
    def generate_explanation(self):
        for type, data_loader in zip(['train', 'valid', 'test'], [self.train_data, self.validation_data, self.test_data]):
            if self.data_type == type:
                self.iterate_items_and_generate_response(
                    data_loader=data_loader,
                    file_save_path=self.file_save_path,
                )

    def generate_image_descriptions(self, image_path, max_new_tokens, regenerate=False, previous_image_descriptions=[]):
        if regenerate is True:
            instruction = read_prompt(f"data/prompts/regenerate_image_descriptions.txt").format(length=len(previous_image_descriptions), previous_generation="\n".join([f"- {x}" for x in previous_image_descriptions]))
        else:
            instruction = read_prompt(f"data/prompts/generate_image_descriptions.txt")
        
        retry_num = 3
        for _ in range(retry_num):
            response = self.send_request(instruction, image_path=image_path, max_new_tokens=max_new_tokens, temperature=self.temperature)
            if response:
                break

        if not response:
            return []
        
        response = re.split(r'-{3,}', response.strip())[0]
        response = response.replace("[Description]:", "").strip()
        response = sentence_tokenize(response)
        response = [x for x in response if len(x.split(' ')) >= 3 and "\n" not in x and "artist's signature" not in x]
        
        if response:
            response = [x for x in response if x.endswith(".") or x.endswith('"')]
        return response


    def generate_implications(self, image_path, caption, descriptive_sentences, max_new_tokens, implications=[]):
        
        def _generate_response(instruction):
            retry_num = 3
            for _ in range(retry_num):
                response = self.send_request(instruction, image_path=image_path, max_new_tokens=max_new_tokens, temperature=self.temperature)
                if response is not None and response:
                    break
            
            if response is None or not response:
                return []

            response = re.split(r'-{3,}', response.strip())[0]
            response = response.replace("[Connections]:", "").strip()
            response = response.replace("Dr.", "DR_DOT")
            return [x for x in sentence_tokenize(response) if len(x.split(" ")) >= 3 and not x.endswith(":") and not x.startswith("Here are") and len(x.strip()) > 10]
        
        goal = ''
        if self.data_name == "memecap":
            goal =  "Your ultimate goal is to generate a concise, specific sentence that explains what the meme poster is trying to convey."
        elif self.data_name == "newyorker":
            goal =  "Your ultimate goal is to generate a specific explanation that describes why the caption is funny for an image."
        elif self.data_name == "yesbut":
            goal =  "Your ultimate goal to generate a specific explanation that describes why the image is funny or satirical."
        
        new_implications = []
        chunk_size = 2
        if not implications:
            for i in range(0, len(descriptive_sentences)):
                chunk_desc = descriptive_sentences[i:min(i+chunk_size, len(descriptive_sentences))]
                if not chunk_desc or len(chunk_desc) < chunk_size:
                    break
                instruction = read_prompt(f"data/prompts/generate_seed_implications.txt")
                instruction = instruction.format(goal=goal, caption=caption, description="\n".join([f"- {x}"for x in chunk_desc]))
                imps = _generate_response(instruction)
                # print("[1st round imps]:\n", "\n".join(f"\t- {x}" for x in imps))
                count = 0
                while len(imps) > 3:
                    print("Regenerating seed implications...")
                    instruction = read_prompt(f"data/prompts/regenerate_seed_implications.txt")
                    instruction = instruction.format(goal=goal, caption=caption, description="\n".join([f"- {x}"for x in chunk_desc]), previous_generation="\n".join([f"- {x}"for x in imps]), length=len(imps))
                    imps = _generate_response(instruction)
                    count += 1
                    if count > 3:
                        break
                    
                new_implications.extend(imps)

        else:
            # The connections can involve multiple logical steps (multihop). 
            for implication in implications:
                for i in range(0, len(descriptive_sentences)):
                    chunk_desc = descriptive_sentences[i:min(i+chunk_size, len(descriptive_sentences))]
                    if not chunk_desc or len(chunk_desc) < chunk_size:
                        break
                    instruction = read_prompt(f"data/prompts/generate_nonseed_implications.txt")
                    instruction = instruction.format(goal=goal, caption=caption, description="\n".join([f"- {x}"for x in chunk_desc]), implication="\n".join([f"- {x}"for x in implication]))
                    imps = _generate_response(instruction)
                    # print("[2nd round imps]:\n", "\n".join(f"\t- {x}" for x in imps))
                    count = 0
                    while len(imps) > 3:
                        print("Regenerating nonseed implications...")
                        instruction = read_prompt(f"data/prompts/regenerate_nonseed_implications.txt")
                        instruction = instruction.format(
                            goal=goal, caption=caption, description="\n".join([f"- {x}"for x in chunk_desc]), implication="\n".join([f"- {x}"for x in implication]), previous_generation="\n".join([f"- {x}"for x in imps]), length=len(imps)
                        )
                        imps = _generate_response(instruction)
                        count += 1
                        if count > 3:
                            break
                    
                    new_implications.extend(imps)
        # filtering out some invalid sentences
        filtered_implications = [
            sent.replace("[Connections]:", "").replace("DOUBLE_QUOTE", '"').replace("SINGLE_QUOTE", "'").replace("DR_DOT", "Dr.").lstrip("-").strip()
            for sent in new_implications
            if (sent.endswith(".") and 
                not any(keyword in sent.lower() for keyword in ["[caption]", "[description]", "[connections]", "without", "overall", "in summary", "for example"]) and 
                len(sent.split(' ')) > 5 and 
                "\n" not in sent and 
                ":" not in sent)
        ]
        return filtered_implications.copy()


    def generate_answer_from_vlm(self, image_path, caption, max_new_tokens, image_descriptions=[], implications=[], candidate_response='', criticism=''):
        """
            when generating candidate response, the candidate_response is empty
            when generating the final response, the candidate_response is not empty
        """
        instruction = get_vlm_response_prompt(
            data_name=self.data_name, caption=caption, image_descriptions=image_descriptions, implications=implications, candidate_response=candidate_response, add_cot=self.add_cot, criticism=criticism
        )
        response = self.send_request(instruction, image_path=image_path, max_new_tokens=max_new_tokens, temperature=self.temperature)
        if "Output:" in response:
            response = response[response.index("Output:"):]
            response = response.replace("Output:", "").strip()
        return (instruction, response.strip())
        
    def run_fast_clustering(self, sentences, sentence_embeddings, min_cluster_size=1, threshold=0.9):
       
        if not sentences:
            return {
                'sentences': [],
                'embeddings': None
            }
                    
        clusters = st_util.community_detection(sentence_embeddings, min_community_size=min_cluster_size, threshold=threshold)
        clustered_sentences = []
        selected_indices = []
        for cluster_indices in clusters:
            # find most representative sentence
            if len(cluster_indices) <= 2:
                # print("[cluster-sentences]:", [sentences[x] for x in cluster_indices])                
                clustered_sentences.append(sentences[cluster_indices[0]])
                selected_indices.append(cluster_indices[0])
            else:
                # choose the sentence that is closer to centroid
                clustered_sentence_embeddings = sentence_embeddings[cluster_indices].cpu().detach().numpy()
                centroid = np.mean(clustered_sentence_embeddings, axis=0)
                similarities = cosine_similarity([centroid], clustered_sentence_embeddings)[0]
                # print("[similarities]:", similarities)
                most_representative_idx = cluster_indices[np.argmax(similarities)]
                clustered_sentences.append(sentences[most_representative_idx])
                selected_indices.append(most_representative_idx)
                # print("[cluster-sentences]:", [sentences[x] for x in cluster_indices])
                # print("[most_representative_sentence]:", sentences[most_representative_idx])
        # cluster_indices = [random.sample(cluster, 1)[0] for cluster in clusters]
        # clustered_sentences = [sentences[x] for x in cluster_indices]
        clustered_sentence_embeddings = sentence_embeddings[selected_indices]
        return {
            'sentences': clustered_sentences,
            'embeddings': clustered_sentence_embeddings
        }

    def find_topk_implications(self, caption, image_descriptions, beams, implications, candidate_responses, weight, beam_size=3, selection_method=""):
        new_beams = []
        # Expand each current beam
        for found_implications, _, _ in beams: # beams (implications, cosine_similarity, cross entropy)
            
            for implication in implications:
                if implication in found_implications:
                    continue
                
                new_implications = found_implications + [implication]
                implication_embedding = get_sentence_transformer_embedding([new_implications[-1]])
                description_embedding = get_sentence_transformer_embedding(image_descriptions+found_implications)                
                max_cos_sim, _ = calculate_cosine_similarity(implication_embedding, description_embedding)
                
                if selection_method == "cosine" or weight == 0:
                    min_cross_entropy_value = 0
                else:
                    input_prompt = get_cross_entropy_response_prompt(self.data_name, caption, image_descriptions=image_descriptions, implications=new_implications)
                    all_cross_entropy_values = calculate_batched_cross_entropy(batch_size=4, input_prompt=input_prompt, targets=candidate_responses, device=self.device)
                    min_cross_entropy_value = min(all_cross_entropy_values)
                new_beams.append((new_implications, max_cos_sim, min_cross_entropy_value))
                # print("\t[]:", (new_implications, max_cos_sim, min_cross_entropy_value),  weight*max_cos_sim + (1-weight)*min_cross_entropy_value)

        if new_beams:
            if selection_method == "ce": # use CE value only
                new_beams = [(x[0], 0, x[2]) for x in new_beams]
            elif selection_method == "cosine": # use CE value only
                new_beams = [(x[0], x[1], 0) for x in new_beams]                
            else:
                new_beams = [(x[0], x[1], weight*x[2]) for x in new_beams]
            new_beams.sort(key=lambda x: x[1]+x[2])
            beams = new_beams[:beam_size]
        best_beam = min(beams, key=lambda x: x[1]+x[2])
        return beams, best_beam
    
    def run_multihop_inference(self, image_path, caption, image_descriptions, candidate_responses, selection_method=""):
        if isinstance(candidate_responses, str):
            candidate_responses = [candidate_responses]

        beam_size = 3
        beams = [([], 0.0, 0.0)] # list of (implications, cross-entropy loss, similarity between caption and image)
        previous_implications = []
        selected_implications = []
        
        def _cluster_implications(implications, threshold):
            # operation with the same set
            # cluster similar implications (i.e. deduplication across the same set of implications).
            embeddings = get_sentence_transformer_embedding(implications)
            clustered = self.run_fast_clustering(
                sentences=implications, sentence_embeddings=embeddings,
                min_cluster_size=1, threshold=threshold
            )
            return clustered['sentences']
        
        selected_beams = []
        all_candidate_responses = [candidate_responses]
        for hop in range(self.num_hops):
            print(f"======================[hop: {hop}]======================")
            # print("[candidate_responses]:\n", "\n".join(candidate_responses))
            
            implications = self.generate_implications(
                image_path=image_path, caption=caption, descriptive_sentences=image_descriptions, implications=selected_implications, max_new_tokens=512
            )
            implications = list(dedupe(implications, threshold=97, scorer=fuzz.ratio))

            # cluster implications.
            if implications:
                if len(implications) > 15:
                    implications = _cluster_implications(implications, threshold=self.cosine_threshold)
                previous_implications.append(implications.copy())
            
            
            # Find top-k implications
            if selection_method == "random":
                selected_implications.append(random.sample(implications, min(3, len(implications))))
            else:
                beams, _ = self.find_topk_implications(
                    caption=caption, image_descriptions=image_descriptions, beams=beams, implications=implications, candidate_responses=candidate_responses, weight=self.weight, beam_size=beam_size, selection_method=selection_method
                )
                selected_beams.append(beams)

                # Select top-k implications (beams contain ([list of implications, cosine similarity, cross entropy value]))
                selected_implications = [x[0] for x in beams] # a list of lists
                print("[selected_implications]:\n", len(selected_implications), selected_implications)
            
            # Add new candidate responses to the set of previous candidate responses.
            new_candidate_responses = []
            for imp in selected_implications:
                if len(candidate_responses) <= 1:
                    selected_ce_response = candidate_responses
                else:
                    input_prompt = get_cross_entropy_response_prompt(self.data_name, caption, image_descriptions=image_descriptions, implications=imp)
                    all_cross_entropy_values = calculate_batched_cross_entropy(batch_size=4, input_prompt=input_prompt, targets=candidate_responses, device=self.device)
                    min_top_1_indices = sorted(range(len(all_cross_entropy_values)), key=lambda i: all_cross_entropy_values[i])[:min(1, len(all_cross_entropy_values))]
                    selected_ce_response = [candidate_responses[x] for x in min_top_1_indices] #candidate_responses[all_cross_entropy_values.index(min(all_cross_entropy_values))]
                
                _, new_candidate_response = self.generate_answer_from_vlm(
                    image_path=image_path, caption=caption, max_new_tokens=512, image_descriptions=image_descriptions, implications=imp, candidate_response="\n".join([f"- {x}"for x in selected_ce_response])
                )                
                if new_candidate_response not in new_candidate_responses:
                    new_candidate_responses.append(new_candidate_response)

            candidate_responses = candidate_responses+new_candidate_responses
            if len(candidate_responses) > 1:
                candidate_responses = list(dedupe(candidate_responses, threshold=97, scorer=fuzz.ratio))

            # find best candidate_responses
            if len(selected_implications[0]) > 0:
                current_implications = sum(selected_implications, [])[0]
                input_prompt = get_cross_entropy_response_prompt(self.data_name, caption, image_descriptions=image_descriptions, implications=current_implications)
            else:
                input_prompt = get_cross_entropy_response_prompt(self.data_name, caption, image_descriptions=image_descriptions, implications=[])
            
            if len(candidate_responses) > 3:
                all_cross_entropy_values = calculate_batched_cross_entropy(batch_size=4, input_prompt=input_prompt, targets=candidate_responses, device=self.device)
                min_top_3_indices = sorted(range(len(all_cross_entropy_values)), key=lambda i: all_cross_entropy_values[i])[:min(3, len(all_cross_entropy_values))]
                best_candidate_responses = [candidate_responses[x] for x in min_top_3_indices]
                candidate_responses = best_candidate_responses
            
            all_candidate_responses.append(candidate_responses)

        return all_candidate_responses, previous_implications, selected_implications, selected_beams


    def self_improve(self, image_path, caption):
        goal = ""
        if self.data_name == "newyorker":
            goal = "You will be given with an image along with its caption, and a candidate response that explains why the caption is funny for the given image."
        elif self.data_name == "memecap":
            goal = "You will be given with a meme along with its caption, and a candidate response that describes what meme poster is trying to convey."
        elif self.data_name == "yesbut":
            goal = "You will be given with an image and a candidate response that describes why the image is funny or satirical."
                            
        _, candidate_response = self.generate_answer_from_vlm(image_path=image_path, caption=caption, max_new_tokens=512, image_descriptions=[], implications=[], candidate_response='')
        candidate_response_with_critic_list = [candidate_response]
        candidate_response_without_critic_list = [candidate_response]
        criticism_list = []
        for i in range(2):
            EVAL_PROMPT = """{goal} Your task is to criticize the candidate response based on the following evaluation criteria:
- Relevance: Does the explanation directly address why the caption is funny, considering both the image and its caption?
- Insightfulness: Does the explanation provide a meaningful and well-reasoned interpretation of the humor?
- Completeness: Does the explanation address all relevant aspects of the caption and image (e.g., visual details, text) that contribute to the humor?
- Accuracy: Is the explanation factually consistent with the details in the image and caption?
- Clarity: Is the explanation clear, concise, and free from unnecessary ambiguity?

Proceed to criticize the candidate response ideally using less than 5 sentences:
"""

            if self.data_name == "newyorker" or self.data_name == "memecap":
                EVAL_PROMPT += """[Caption]: {caption}

[Candidate Response]:
{candidate_response}

[Output]:"""
            elif self.data_name == "yesbut":
                EVAL_PROMPT += """[Candidate Response]:
{candidate_response}

[Output]:"""

            if self.data_name == "yesbut":
                criticism = self.send_request(EVAL_PROMPT.format(goal=goal, candidate_response=candidate_response), image_path=image_path, max_new_tokens=512, temperature=self.temperature)
            else:
                criticism = self.send_request(EVAL_PROMPT.format(goal=goal, caption=caption, candidate_response=candidate_response), image_path=image_path, max_new_tokens=512, temperature=self.temperature)
            
            _, candidate_response_with_critic = self.generate_answer_from_vlm(image_path=image_path, caption=caption, max_new_tokens=512, image_descriptions=[], implications=[], candidate_response=candidate_response, criticism=criticism)
            _, candidate_response_without_critic = self.generate_answer_from_vlm(image_path=image_path, caption=caption, max_new_tokens=512, image_descriptions=[], implications=[], candidate_response=candidate_response, criticism='')
            criticism_list.append(criticism)
            candidate_response_with_critic_list.append(candidate_response_with_critic)
            candidate_response_without_critic_list.append(candidate_response_without_critic)
        return criticism_list, candidate_response_with_critic_list, candidate_response_without_critic_list

    
    # responses
    def iterate_items_and_generate_response(self, data_loader, file_save_path):
        items = []
        existing_data = {}
        if os.path.exists(file_save_path) and not self.overwrite:
            jsonfile = open(file_save_path, 'rb')
            items = json.load(jsonfile)
            for x in items:
                if 'model_response-cot' in x and x['model_response-cot']:
                    existing_data[os.path.basename(x['image_file_path'])] = x['model_response-cot']
                if 'candidate_responses_with_critic' in x and x['candidate_responses_with_critic']:
                    existing_data[os.path.basename(x['image_file_path'])] = x['candidate_responses_with_critic']

        count = -1
        for item in tqdm.tqdm(data_loader[:110], total=len(data_loader)):
            count += 1
            print("Start generation...")
            image_path, reference, caption = item['local_image_path'], item['reference'], item['caption']
            image_name = os.path.basename(image_path)
            if image_name in existing_data:
                continue

            print("[img_path]:", os.path.abspath(image_path), "[caption]:", caption)
            
            if self.self_improve:
                criticism_list, candidate_response_with_critic_list, candidate_response_without_critic_list = self.self_improve(image_path, caption)
                
                output_data = {
                    'image_file_path': image_path,
                    'caption': caption,
                    'reference': reference,
                    'criticisms': criticism_list,
                    'candidate_responses_with_critic': candidate_response_with_critic_list,
                    'candidate_responses_without_critic': candidate_response_without_critic_list,
                }
                items.append(output_data)
                
                print(f"Saving to: {file_save_path}")
                with open(file_save_path, "wb") as f:
                    f.write(json.dumps(items, indent=2, ensure_ascii=False).encode("utf-8"))
                
                continue

            out_responses = {}
            selected_ce_response = ""
            final_instruction = ""
            all_implications, selected_implications = [], []
            all_candidate_responses, candidate_responses = [], []
            retry_num = 3

            for _ in range(retry_num):
                _, candidate_response = self.generate_answer_from_vlm(image_path=image_path, caption=caption, max_new_tokens=512, image_descriptions=[], implications=[], candidate_response='')                    
                if candidate_response:
                    out_responses['model_response-none'] = candidate_response
                    candidate_responses.append(candidate_response)
                    break

            if self.add_cot:
                for _ in range(retry_num):
                    original_add_cot = self.add_cot
                    self.add_cot = True
                    _, candidate_response = self.generate_answer_from_vlm(image_path=image_path, caption=caption, max_new_tokens=512, image_descriptions=[], implications=[], candidate_response='')                    
                    reasoning, candidate_response = parse_cot_response(candidate_response)
                    self.add_cot = original_add_cot
                    if candidate_response:
                        out_responses['model_reasoning-cot'] = reasoning
                        out_responses['model_response-cot'] = candidate_response
                        break
            
            if self.add_image_descriptions:            
                image_descriptions = []
                if self.add_image_descriptions or self.add_implications:
                    image_descriptions = self.generate_image_descriptions(image_path=image_path, max_new_tokens=128)
                    count = 0
                    while len(image_descriptions) > 5:
                        image_descriptions = self.generate_image_descriptions(image_path=image_path, max_new_tokens=128, regenerate=True, previous_image_descriptions=image_descriptions)
                        # print("[regenerated_image_descriptions]:\n", len(image_descriptions), "\n".join(image_descriptions))
                        count += 1
                        if count > 5:
                            break
                    print("[image_descriptions]:\n", len(image_descriptions), "\n".join(image_descriptions))
                    
                if not image_descriptions or len(image_descriptions) > 5: # takes too long time to inference implications
                    continue

                for _ in range(retry_num):
                    _, candidate_response = self.generate_answer_from_vlm(image_path=image_path, caption=caption, max_new_tokens=512, image_descriptions=image_descriptions, implications=[], candidate_response='')
                    out_responses['model_response-desc'] = candidate_response
                    if candidate_response:
                        break

            if self.add_implications:
                if not candidate_responses or (self.add_image_descriptions and (not image_descriptions)):
                    continue
                all_candidate_responses, all_implications, selected_implications, beams = self.run_multihop_inference(image_path, caption, image_descriptions, candidate_responses, selection_method=self.selection_method)
                
                if not selected_implications:
                    continue
                selected_implications = selected_implications
                selected_ce_response = all_candidate_responses[-1]
                
                selected_implications = list(dedupe(list(OrderedDict.fromkeys(sum(selected_implications, []))), threshold=97, scorer=fuzz.ratio))
                final_instruction, final_response = self.generate_answer_from_vlm(image_path=image_path, caption=caption, max_new_tokens=512, image_descriptions=image_descriptions, implications=selected_implications, candidate_response='')
                out_responses[f'model_response-desc_imp'] = final_response

                final_instruction, final_response = self.generate_answer_from_vlm(image_path=image_path, caption=caption, max_new_tokens=512, image_descriptions=[], implications=selected_implications, candidate_response='')
                out_responses[f'model_response-imp'] = final_response
                
                final_instruction, final_response = self.generate_answer_from_vlm(image_path=image_path, caption=caption, max_new_tokens=512, image_descriptions=image_descriptions, implications=[], candidate_response="\n".join([f"- {x}"for x in selected_ce_response]))
                out_responses[f'model_response-desc_cand'] = final_response

                final_instruction, final_response = self.generate_answer_from_vlm(image_path=image_path, caption=caption, max_new_tokens=512, image_descriptions=[], implications=[], candidate_response="\n".join([f"- {x}"for x in selected_ce_response]))
                out_responses[f'model_response-cand'] = final_response
                
                final_instruction, final_response = self.generate_answer_from_vlm(image_path=image_path, caption=caption, max_new_tokens=512, image_descriptions=image_descriptions, implications=selected_implications, candidate_response="\n".join([f"- {x}"for x in selected_ce_response]))
                out_responses[f'model_response-desc_imp_cand'] = final_response

                final_instruction, final_response = self.generate_answer_from_vlm(image_path=image_path, caption=caption, max_new_tokens=512, image_descriptions=[], implications=selected_implications, candidate_response="\n".join([f"- {x}"for x in selected_ce_response]))
                out_responses[f'model_response-imp_cand'] = final_response

            print("[final_responses]:")
            for k, v in out_responses.items():
                print(f"[{k}]:\n", v)
            output_data = {
                'image_file_path': image_path,
                'caption': caption,
                'reference': reference,
                'model_image_descriptions': image_descriptions,
                'model_implications': all_implications,
                'model_selected_implications': selected_implications,
                'model_all_candidate_responses': all_candidate_responses,
                'model_selected_candidate_responses': candidate_responses,
            }
            output_data.update(out_responses)
            items.append(output_data)
            
            print(f"Saving to: {file_save_path}")
            with open(file_save_path, "wb") as f:
                f.write(json.dumps(items, indent=2, ensure_ascii=False).encode("utf-8"))
            # time.sleep(10)
