import argparse
import collections
import json
import os
import random
import time

import google.generativeai as genai
import tqdm

# Global configuration
random.seed(42)
GEMINI_VERSION = "gemini-1.5-flash"
genai.configure(api_key=os.environ['GEMINI_KEY'])


class PrecisionEvaluator:
    # Prompts for decomposition and evaluation
    FACT_DECOM_PROMPT = (
        "You will be given a paragraph ([Paragraph]). Please break down the [Paragraph] into a stringified Python list of atomic sentences.\n\n"
        "Here are some guidelines when generating atomic sentences:\n"
        "* Do not alter or paraphrase details in the original [Paragraph].\n"
        "* Avoid using pronouns. Be specific when referring to objects, characters or situations.\n"
        "* Do not include too vague sentences (e.g. \"Meme poster is trying to convey a message.\", \"The image is funny.\", etc)\n\n"
        "Here is an example:\n\n"
        "--------\n\n"
        "[Paragraph]:\n"
        "The caption, \"This is the most advanced case of Surrealism I've seen.\", is funny because it humorously treats the surreal and impossible scene—where a person is divided into separate body parts—as a diagnosable medical condition. It playfully applies the term \"Surrealism,\" an art style known for bizarre, dreamlike imagery, to a clinical context. The contrast between the doctor’s serious tone and the absurd situation creates comedic irony.\n\n"
        "[Output]:\n"
        "```python\n"
        "[\n"
        "    \"The scene is surreal.\",\n"
        "    \"The scene is impossible.\",\n"
        "    \"The scene shows a person divided into separate body parts.\",\n"
        "    \"The division of body parts is presented as a diagnosis of a medical condition.\",\n"
        "    \"Surrealism is an art style known for bizarre and dreamlike imagery.\",\n"
        "    \"The doctor has a serious tone.\",\n"
        "    \"The situation is absurd.\",\n"
        "    \"The contrast between the doctor's serious tone and the absurd situation creates comedic irony.\"\n"
        "]\n"
        "```\n\n"
        "--------\n\n"
        "Proceed to break down the following paragraph into a list of atomic sentences.\n"
        "[Paragraph]:\n"
        "{paragraph}\n\n"
        "[Output]:"
    )

    FACT_INFER_PROMPT = (
        "Your task is to assess whether [Sentence1] is inferable from [Sentence2]. [Sentence2] may consist of multiple sentences.\n\n"
        "Here are the evaluation guidelines:\n"
        "1. Mark \"Yes\" if [Sentence1] can be inferred from [Sentence2] — whether explicitly stated, implicitly conveyed, reworded, or serving as supporting information.\n"
        "2. Mark 'No' if [Sentence1] is absent from [Sentence2], cannot be inferred, or contradicts it.\n\n"
        "Proceed to evaluate. \n\n"
        "[Sentence1]: {sentence1}\n\n"
        "[Sentence2]: {sentence2}\n\n"
        "[Output]:"
    )

    FACT_MATCH_PROMPT = (
        "Your task is to assess whether the information in [Sentence1] is present in [Sentence2]. [Sentence2] may consist of multiple sentences.\n\n"
        "Here are the evaluation guidelines:\n"
        "1. Mark 'Yes' if the information in [Sentence1] is present in [Sentence2].\n"
        "2. Mark 'No' if [Sentence2] does not contain the information in [Sentence1].\n\n"
        "Proceed to evaluate. \n\n"
        "[Sentence1]: {sentence1}\n\n"
        "[Sentence2]: {sentence2}\n\n"
        "[Output]:"
    )

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.gemini = genai.GenerativeModel(GEMINI_VERSION)
        # Cache responses to avoid duplicate API calls.
        self.response_to_answer = {}
        self.fact_proportion_dict = collections.defaultdict(list)

    def send_request(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Send a Gemini request with up to 20 retries."""
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

    def decompose_paragraph(self, paragraph: str) -> list:
        """
        Decompose a paragraph into atomic sentences using FACT_DECOM_PROMPT.
        Attempts to eval the returned text; falls back to line-splitting.
        """
        prompt = self.FACT_DECOM_PROMPT.format(paragraph=paragraph)
        response = self.send_request(prompt=prompt, temperature=0.2, max_tokens=512)
        if not response:
            return []
        try:
            cleaned = response.replace("```python", "").replace("```", "").strip()
            decomposed = eval(cleaned)
        except Exception:
            decomposed = [line.strip() for line in response.split("\n") if line.strip().startswith('"')]
        return decomposed

    def evaluate_entailment(self, facts: list, reference: str, prompt_template: str) -> (list, int):
        """
        For each fact, assess whether it is inferable/present in reference using the given prompt.
        Returns a list of answers and the count of 'yes' responses.
        """
        answer_list = []
        yes_count = 0

        def _parse(response_text: str):
            lines = [x.strip() for x in response_text.split("\n") if x.strip()]
            output = lines[0].replace(".", "")
            reasoning = "\n".join(lines)
            return output, reasoning

        for fact in facts:
            formatted_prompt = prompt_template.format(sentence1=fact, sentence2=reference)
            response = self.send_request(prompt=formatted_prompt, temperature=0.2, max_tokens=3)
            if response:
                output, _ = _parse(response)
                if "yes" in output.lower():
                    yes_count += 1
                answer_list.append(output)
            else:
                answer_list.append('')
        return answer_list, yes_count

    def inner_evaluate(self, response_list: list, image_descriptions: str, reference: str) -> dict:
        """
        For each (response, label) tuple, decompose the response into atomic sentences,
        evaluate entailment, and if needed, adjust using image descriptions.
        Returns a dictionary mapping response labels to evaluation outputs.
        """
        output = {}
        for response, resp_label in response_list:
            if not response.strip():
                continue

            print("------" * 10)
            print(f"[{resp_label}]:", response)
            # Decompose the model response into atomic sentences.
            fact_decom_pred = self.decompose_paragraph(response)
            if not fact_decom_pred:
                continue

            if response in self.response_to_answer:
                cached = self.response_to_answer[response]
                answer_list, yes_count = cached["answer_list"], cached["yes_count"]
            else:
                answer_list, yes_count = self.evaluate_entailment(
                    facts=fact_decom_pred,
                    reference=reference,
                    prompt_template=self.FACT_INFER_PROMPT,
                )
            if not answer_list or len(answer_list) != len(fact_decom_pred):
                print(f"Length mismatch!: {len(answer_list)} vs {len(fact_decom_pred)}")
                continue

            # For facts marked 'no', if image descriptions exist, check if they are present there.
            no_indices = [i for i, ans in enumerate(answer_list) if 'no' in ans.lower().strip()]
            no_facts = [fact_decom_pred[i] for i in no_indices]
            desc_answers = []
            if no_facts and image_descriptions:
                desc_answers, _ = self.evaluate_entailment(
                    facts=no_facts,
                    reference=image_descriptions,
                    prompt_template=self.FACT_MATCH_PROMPT,
                )
            if len(desc_answers) == len(no_facts):
                for idx, ans in enumerate(desc_answers):
                    if ans.strip().lower() == 'yes':
                        answer_list[no_indices[idx]] = 'yes-desc'
                        yes_count += 1

            # Cache the evaluation result.
            if response not in self.response_to_answer:
                self.response_to_answer[response] = {"answer_list": answer_list, "yes_count": yes_count}

            for fact, ans in zip(fact_decom_pred, answer_list):
                print(f"[{ans}]: {fact}")

            print("[score]:", yes_count / len(fact_decom_pred), "[yes]:", yes_count, "[total]:", len(fact_decom_pred))
            self.fact_proportion_dict[resp_label].append(yes_count / len(fact_decom_pred))
            output[f'fact_decom_pred-{resp_label}'] = fact_decom_pred
            output[f'model_output-{resp_label}'] = answer_list
        return output

    def extract_responses(self, item: dict, mode: str):
        """
        Extract response(s) from the item based on the evaluation mode.
        Modes:
          - "default": Uses fixed response keys.
          - "candidate": Uses candidate responses from model_all_candidate_responses plus model_response-imp_cand.
          - "self_improve": Uses the last candidate response from both with and without critic.
        Returns (response_list, extra_data) where response_list is a list of (response, label) tuples.
        """
        response_list = []
        extra_data = {}
        if mode == "default":
            required = [
                'model_response-none', 'model_response-cot', 'model_response-desc',
                'model_response-desc_imp', 'model_response-desc_cand', 'model_response-desc_imp_cand',
                'model_response-imp', 'model_response-cand', 'model_response-imp_cand'
            ]
            if not all(k in item for k in required):
                return None, None
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
            if "model_all_candidate_responses" not in item or not item.get("model_all_candidate_responses"):
                return None, None
            if "model_response-imp_cand" not in item or not item.get("model_response-imp_cand", "").strip():
                return None, None
            all_candidates = list(set(sum(item["model_all_candidate_responses"], [])))
            all_candidates = [r for r in all_candidates if r.strip()]
            resp_to_id = {resp: f"candidate_response_{idx}" for idx, resp in enumerate(all_candidates)}
            for resp in all_candidates:
                response_list.append((resp, resp_to_id[resp]))
            response_list.append((item["model_response-imp_cand"], "response_imp_cand"))
            extra_data["candidate_original"] = item["model_all_candidate_responses"]
            extra_data["resp_to_id"] = resp_to_id
        elif mode == "self_improve":
            if ("candidate_responses_with_critic" not in item or not item.get("candidate_responses_with_critic")) or \
               ("candidate_responses_without_critic" not in item or not item.get("candidate_responses_without_critic")):
                return None, None
            response_with = item["candidate_responses_with_critic"][-1]
            response_without = item["candidate_responses_without_critic"][-1]
            response_list.append((response_with, "response_with_critic"))
            response_list.append((response_without, "response_without_critic"))
        else:
            return None, None

        return response_list, extra_data

    def evaluate_generic(self, data: list, out_file: str, existing_image_paths: set, result: list,
                         assist_data: dict, mode: str) -> (list, dict):
        """
        Generic evaluation function for processing each item.
        Builds image descriptions (from model_image_descriptions plus caption),
        extracts responses (based on mode), runs inner evaluation, and updates the record.
        """
        for item in tqdm.tqdm(data, total=len(data)):
            image_path = item.get("image_file_path", "")
            image_name = os.path.basename(image_path)
            if image_name in existing_image_paths:
                print(f"Skipping {image_name}...")
                continue

            # Build image description string
            descs = item.get("model_image_descriptions", [])
            caption = item.get("caption", "")
            descs.append(f'The caption says, "{caption}".')
            desc_text = "\n".join([f"- {x}" for x in descs])
            print("[image_descriptions]:", desc_text)
            decomposed_desc = self.decompose_paragraph(desc_text)
            decomposed_desc_str = "\n".join([f"- {x}" for x in decomposed_desc])
            print("[image_descriptions-decomposed]:", decomposed_desc_str)

            # Build reference string
            ref = item.get("reference", "")
            if isinstance(ref, list):
                reference = "\n".join([f"- {x}" for x in ref])
            else:
                reference = ref
            print("[reference]:", reference)

            responses, extra_data = self.extract_responses(item, mode)
            if responses is None:
                continue

            out_data = self.inner_evaluate(responses, decomposed_desc_str, reference)
            extra_out = {}
            if mode == "candidate" and extra_data.get("candidate_original"):
                resp_to_id = extra_data["resp_to_id"]
                for hop, cand_list in enumerate(item["model_all_candidate_responses"]):
                    for idx, resp in enumerate(cand_list):
                        if not resp.strip():
                            continue
                        key = f"model_output-hop{hop}-{idx}th"
                        candidate_label = resp_to_id.get(resp, "")
                        extra_out[key] = out_data.get(f"model_output-{candidate_label}", [])
            # Merge outputs into the item record.
            new_item = {k: item[k] for k in item if k.startswith("model_") or k in ["image_file_path", "caption", "reference", "beams"]}
            new_item.update(out_data)
            new_item.update(extra_out)
            result.append(new_item)

            # Print intermediate fact proportion results.
            for k, v in self.fact_proportion_dict.items():
                avg = sum(v) / len(v) if v else 0
                print(f"[{k}-proportion]:", avg)

            self.save_results(out_file, result)
        return result, self.fact_proportion_dict

    def save_results(self, out_file: str, result: list):
        """Save the result to a JSON file."""
        with open(out_file, 'w') as outfile:
            json.dump(result, outfile, indent=4)
        print("\nFinal result file is saved here:", os.path.abspath(out_file))


def main(args):
    data_name = args.data_name          # e.g., "memecap", "newyorker", "yesbut"
    model_name = args.model_name        # e.g., "gpt4o", "llava-v1.6-34b-hf", etc.
    exp_name = args.exp_name
    seed = args.seed
    weight = args.weight
    selection_method = args.selection_method
    hop = args.hop

    assist_file = ""
    if exp_name == "self-improve":
        in_file = f"data/{data_name}/model-output/{model_name}/{exp_name}/model_output_test_t0.8_h2_s0_sd{seed}.json"
        assist_file = f"data/{data_name}/model-output/{model_name}/desc-imp-cand/final-v1/model_output_test_t0.8_w0.7_h2_s0_sd{seed}.json"
    else:
        in_file = f"data/{data_name}/model-output/{model_name}/desc-imp-cand/{exp_name}/model_output_test_t0.8_w{weight}_h{hop}_s0_sd{seed}{selection_method}.json"

    out_dir = f"evaluation/results/gemini/pred_checking_eval/{GEMINI_VERSION}/{exp_name}/"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{data_name}_{model_name}_" + os.path.basename(in_file).replace(".json", "_eval_precision.json"))
    result = []
    if os.path.exists(out_file) and not args.overwrite:
        result = json.load(open(out_file, 'rb'))

    data = json.load(open(in_file, 'rb'))
    assist_data = {}
    if assist_file:
        assist_json = json.load(open(assist_file, 'rb'))
        for x in assist_json:
            assist_data[os.path.basename(x['image_file_path'])] = x.get('model_image_descriptions', [])

    existing_image_paths = {os.path.basename(x['image_file_path']) for x in result if "image_file_path" in x}

    # (Optional) Analysis mode: aggregate existing evaluation records.
    if (result and not args.overwrite) or args.analysis:
        print("[in_file]:", in_file)
        print("[out_file]:", out_file)
        print("Existing data:", len(data), "Existing eval records:", len(result))
        # [Aggregation code can be inserted here if needed.]

    # Choose evaluation mode based on experiment name.
    if exp_name == "self-improve":
        mode = "self_improve"
    elif exp_name == "analysis":
        mode = "candidate"
    else:
        mode = "default"

    evaluator = PrecisionEvaluator(seed=seed)
    if not args.analysis:
        if mode == "self_improve":
            result, fact_proportion_dict = evaluator.evaluate_generic(data, out_file, existing_image_paths, result, assist_data, mode)
        elif mode == "candidate":
            result, fact_proportion_dict = evaluator.evaluate_generic(data[:50], out_file, existing_image_paths, result, assist_data, mode)
        else:
            result, fact_proportion_dict = evaluator.evaluate_generic(data, out_file, existing_image_paths, result, assist_data, mode)
        for k, v in fact_proportion_dict.items():
            avg = sum(v) / len(v) if v else 0
            print(f"[{k}-proportion]:", avg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='memecap', choices=['memecap', 'newyorker', 'yesbut'], type=str, help='Data directory path')
    parser.add_argument('--model_name', default='gpt4o', type=str, choices=['llava-v1.6-34b-hf', 'gpt4o', 'gemini', 'qwen2', 'phi'], help='Model type to generate responses')
    parser.add_argument('--exp_name', default='final', type=str, help='Experiment name')
    parser.add_argument('--selection_method', default='', type=str, help='Method used to select implications')
    parser.add_argument('--seed', default=0, type=int, help='Seed number')
    parser.add_argument('--weight', default=0.7, type=float, help='Weight parameter')
    parser.add_argument('--hop', default=2, type=int, help='Number of hops')
    parser.add_argument('--overwrite', action='store_true', help="Overwrite file")
    parser.add_argument('--analysis', action='store_true', help="Analysis mode")
    
    args = parser.parse_args()
    main(args)
