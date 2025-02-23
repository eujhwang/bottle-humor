import argparse
import collections
import json
import os
import random
import time
from datetime import datetime

import google.generativeai as genai
import pytz
import tqdm

# Global configuration
random.seed(42)
GEMINI_VERSION = "gemini-1.5-flash"
genai.configure(api_key=os.environ['GEMINI_KEY'])


class FactEvaluator:
    """
    A class that encapsulates logic for sending Gemini requests and evaluating fact entailment.
    Merges similar evaluation functions into one generic function.
    """
    FACT_MATCH_PROMPT = (
        "Your task is to assess whether [Sentence1] is conveyed in [Sentence2]. "
        "[Sentence2] may consist of multiple sentences.\n\n"
        "Here are the evaluation guidelines:\n"
        "1. Mark 'Yes' if [Sentence1] is conveyed in [Sentence2].\n"
        "2. Mark 'No' if [Sentence2] does not convey the information in [Sentence1].\n\n"
        "Proceed to evaluate.\n\n"
        "[Sentence1]: {reference}\n\n"
        "[Sentence2]: {predicted_sentence}\n\n"
        "[Output]:"
    )

    def __init__(self, gemini_version: str = GEMINI_VERSION, seed: int = 42):
        random.seed(seed)
        self.gemini_version = gemini_version
        self.gemini = genai.GenerativeModel(gemini_version)
        # Cache mapping a response to its evaluated answers to avoid repeated API calls.
        self.response_to_answer = {}
        # To store fact proportions for different response types.
        self.fact_proportion_dict = collections.defaultdict(list)

    def send_request(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Sends a Gemini request with retries."""
        for _ in range(20):
            try:
                response = self.gemini.generate_content(
                    [prompt],
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                if response:
                    return response.text
            except Exception as e:
                print(e)
                time.sleep(20)
        return ''

    def evaluate_entailment(self, ref_facts: list, pred_facts: str) -> (list, int):
        """
        For each reference fact in ref_facts, send a Gemini prompt with pred_facts
        to decide whether it is conveyed.
        Returns a list of outputs (answers) and a count of 'Yes' answers.
        """
        answer_list = []
        yes_count = 0

        def _parse_response(response: str) -> (str, str):
            lines = [x.strip() for x in response.split("\n") if x.strip()]
            # Use the first line as the output (removing any trailing period)
            output = lines[0].replace(".", "")
            reasoning = "\n".join(lines)
            return output, reasoning

        for ref in ref_facts:
            prompt = self.FACT_MATCH_PROMPT.format(reference=ref, predicted_sentence=pred_facts)
            response = self.send_request(prompt=prompt, temperature=0.2, max_tokens=3)
            if response:
                output, _ = _parse_response(response)
                answer = output if output else ''
                if "yes" in output.lower():
                    yes_count += 1
                answer_list.append(answer)
        return answer_list, yes_count

    def inner_evaluate(self, response_list: list, decomposed_reference: list) -> dict:
        """
        For each (response, label) in response_list, either use a cached result or call
        evaluate_entailment. Then, update the fact proportions and return an output dict.
        """
        out = {"fact_decom_ref": decomposed_reference}
        for response, response_name in response_list:
            if not response.strip():
                continue

            print("------" * 10)
            print(f"[{response_name}]:", response)

            if response in self.response_to_answer:
                data = self.response_to_answer[response]
                answer_list, yes_count = data["answer_list"], data["yes_count"]
            else:
                answer_list, yes_count = self.evaluate_entailment(
                    ref_facts=decomposed_reference, pred_facts=response
                )
                self.response_to_answer[response] = {"answer_list": answer_list, "yes_count": yes_count}

            if not answer_list:
                continue

            if len(answer_list) != len(decomposed_reference):
                print(
                    f"Length mismatch!: [len(answer_list)]: {len(answer_list)} and "
                    f"[len(decomposed_reference)]: {len(decomposed_reference)}"
                )
                continue

            for ref, ans in zip(decomposed_reference, answer_list):
                print(f"[{ans}]: {ref}")

            fact_proportion = yes_count / len(decomposed_reference)
            self.fact_proportion_dict[response_name].append(fact_proportion)
            out[f"model_output-{response_name}"] = answer_list
        return out

    def extract_responses(self, item: dict, mode: str) -> (list, dict):
        """
        Extracts the list of responses and any extra output data based on the evaluation mode.
        Modes:
          - "default": Uses fixed keys for single-response evaluation.
          - "candidate": Uses candidate responses from 'model_all_candidate_responses' and also includes 'model_response-imp_cand'.
          - "self_improve": Uses the last candidate response from both with and without critic.
        Returns (response_list, extra_data). extra_data can be empty or include keys that need further processing.
        """
        response_list = []
        extra_data = {}
        if mode == "default":
            # Require a set of keys to exist; otherwise, skip this item.
            required_keys = [
                'model_response-none', 'model_response-cot', 'model_response-desc',
                'model_response-desc_imp', 'model_response-desc_cand', 'model_response-desc_imp_cand'
            ]
            if not all(k in item for k in required_keys):
                return None, None
            # Build a list of (response, label) tuples.
            mapping = {
                "response_none": "model_response-none",
                "response_cot": "model_response-cot",
                "response_desc": "model_response-desc",
                "response_desc_imp": "model_response-desc_imp",
                "response_desc_cand": "model_response-desc_cand",
                "response_desc_imp_cand": "model_response-desc_imp_cand",
                "response_imp": "model_response-imp",
                "response_cand": "model_response-cand",
                "response_imp_cand": "model_response-imp_cand",
            }
            for label, key in mapping.items():
                response_list.append((item.get(key, ""), label))
        elif mode == "candidate":
            # Use candidate responses if available.
            if "model_response-imp_cand" not in item or not item.get("model_response-imp_cand", "").strip():
                return None, None
            if "model_all_candidate_responses" not in item or not item.get("model_all_candidate_responses"):
                return None, None
            # Flatten candidate responses (list of lists), deduplicate and filter empty strings.
            all_candidates = list(set(sum(item["model_all_candidate_responses"], [])))
            all_candidates = [resp for resp in all_candidates if resp.strip()]
            # Map each candidate to a unique label.
            resp_to_id = {resp: f"candidate_response_{idx}" for idx, resp in enumerate(all_candidates)}
            for resp in all_candidates:
                response_list.append((resp, resp_to_id[resp]))
            # Also add the response_imp_cand.
            response_list.append((item["model_response-imp_cand"], "response_imp_cand"))
            # Pass along the original candidate responses for later merging.
            extra_data["candidate_original"] = item["model_all_candidate_responses"]
            extra_data["resp_to_id"] = resp_to_id
        elif mode == "self_improve":
            # For self-improve, expect candidate responses with and without critic.
            if ("candidate_responses_with_critic" not in item or not item.get("candidate_responses_with_critic")) or \
               ("candidate_responses_without_critic" not in item or not item.get("candidate_responses_without_critic")):
                return None, None
            response_with = item["candidate_responses_with_critic"][-1]
            response_without = item["candidate_responses_without_critic"][-1]
            response_list.append((response_with, "response_with_critic"))
            response_list.append((response_without, "response_without_critic"))
        else:
            # Unsupported mode.
            return None, None

        return response_list, extra_data

    def evaluate_generic(self, data: list, out_file: str, existing_image_paths: set, result: list,
                         image_path_to_decomposed: dict, mode: str) -> (list, dict):
        """
        A generic evaluation function that processes each item in data based on the evaluation mode.
        It extracts responses, runs inner evaluation, merges extra data if needed,
        and appends the new record to result.
        """
        for item in tqdm.tqdm(data, total=len(data)):
            image_path = item.get("image_file_path", "")
            image_name = os.path.basename(image_path)
            if image_name in existing_image_paths:
                print(f"Skipping {image_name}...")
                continue
            if image_name not in image_path_to_decomposed:
                print(f"Decomposition data doesn't exist. Skipping {image_name}...")
                continue

            fact_decom_ref = image_path_to_decomposed.get(image_name, [])
            if not fact_decom_ref:
                continue

            print("--------" * 10)
            print("[image_path]:", image_path)
            print("[reference]:", item.get("reference", ""))

            responses, extra_data = self.extract_responses(item, mode)
            if responses is None:
                continue

            out_data = self.inner_evaluate(responses, fact_decom_ref)

            # If in candidate mode, further process candidate responses by mapping hop-level outputs.
            extra_out = {}
            if mode == "candidate" and "candidate_original" in extra_data:
                resp_to_id = extra_data["resp_to_id"]
                # For each hop, map candidate responses to corresponding evaluation outputs.
                for hop, candidate_list in enumerate(item["model_all_candidate_responses"]):
                    for idx, resp in enumerate(candidate_list):
                        if not resp.strip():
                            continue
                        key = f"model_output-hop{hop}-{idx}th"
                        # Use the unique label from resp_to_id to retrieve the output.
                        candidate_label = resp_to_id.get(resp, "")
                        extra_out[key] = out_data.get(f"model_output-{candidate_label}", [])
            
            # Build a new record: include common fields plus all model outputs.
            new_item = {k: item[k] for k in item if k.startswith("model_") or k in ["image_file_path", "caption", "reference", "beams"]}
            new_item.update(out_data)
            new_item.update(extra_out)
            result.append(new_item)

            # Display intermediate fact proportion results.
            for k, v in self.fact_proportion_dict.items():
                avg = sum(v) / len(v) if v else 0
                print(f"[{k}-proportion]:", avg)

            # Save intermediate results.
            self.save_results(out_file, result)

        return result, self.fact_proportion_dict

    def save_results(self, out_file: str, result: list):
        """Save results to a JSON file."""
        with open(out_file, 'w') as outfile:
            json.dump(result, outfile, indent=4)
        print("\nFinal result file is saved here:", os.path.abspath(out_file))


def main(args):
    data_name = args.data_name           # e.g., "memecap", "newyorker", "yesbut"
    model_name = args.model_name         # e.g., "gpt4o", "llava-v1.6-34b-hf", etc.
    weight = args.weight
    exp_name = args.exp_name
    seed = args.seed
    selection_method = args.selection_method
    hops = args.hop

    # Determine input file paths based on experiment name.
    if exp_name == "self-improve":
        in_file = f"data/{data_name}/model-output/{model_name}/{exp_name}/model_output_test_t0.8_h{hops}_s0_sd{seed}.json"
    else:
        in_file = f"data/{data_name}/model-output/{model_name}/desc-imp-cand/{exp_name}/model_output_test_t0.8_w{weight}_h{hops}_s0_sd{seed}{selection_method}.json"

    decomp_file = f"evaluation/results/gemini/fact_decomposed/seed0/{data_name}_reference_decomposed.json"
    out_dir = f"evaluation/results/gemini/{GEMINI_VERSION}/{exp_name}/"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir,
        f"{data_name}_{model_name}_" + os.path.basename(in_file).replace(".json", "_eval_recall.json")
    )

    # Load existing result if available and not overwriting.
    result = []
    if os.path.exists(out_file) and not args.overwrite:
        result = json.load(open(out_file, 'rb'))

    data = json.load(open(in_file, 'rb'))
    decomposition_data = json.load(open(decomp_file, 'rb'))
    image_path_to_decomposed = {
        os.path.basename(x['image_file_path']): x['decomposed_reference']
        for x in decomposition_data
    }
    existing_image_paths = {os.path.basename(x['image_file_path']) for x in result if "image_file_path" in x}

    evaluator = FactEvaluator(seed=seed)

    # Choose evaluation mode based on experiment name.
    if exp_name == "self-improve":
        mode = "self_improve"
    elif exp_name == "analysis":
        mode = "candidate"
    else:
        mode = "default"

    evaluator.evaluate_generic(data, out_file, existing_image_paths, result, image_path_to_decomposed, mode)

    # Optionally, print aggregated fact proportions.
    for k, v in evaluator.fact_proportion_dict.items():
        avg = sum(v) / len(v) if v else 0
        print(f"[{k}-proportion]:", avg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='memecap', choices=['memecap', 'newyorker', 'yesbut'], type=str, help='Data directory path')
    parser.add_argument('--model_name', default='gpt4o', type=str, choices=['gpt4o', 'gemini', 'qwen2', 'phi'], help='Model type to generate responses')
    parser.add_argument('--exp_name', default='final', type=str, help='Experiment name')
    parser.add_argument('--selection_method', default='', type=str, help='Method used to select implications')
    parser.add_argument('--seed', default=42, type=int, help='Seed number')
    parser.add_argument('--weight', default=0.7, type=float, help='Temperature weight')
    parser.add_argument('--hop', default=2, type=int, help='Number of hops')
    parser.add_argument('--overwrite', action='store_true', help="Overwrite file")
    parser.add_argument('--analysis', action='store_true', help="Analysis mode")
    
    args = parser.parse_args()
    main(args)
