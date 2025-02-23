import argparse
import json
import os
import random
import time
import tqdm
import google.generativeai as genai

random.seed(42)


class FactDecomposer:
    """
    Uses Gemini to decompose a paragraph into a list of atomic sentences.
    """

    FACT_DECOM_PROMPT = (
        "You will be given a paragraph ([Paragraph]). Please break down the [Paragraph] into a stringified Python list of atomic sentences.\n\n"
        "Here are some guidelines when generating atomic sentences:\n"
        "* Do not alter or paraphrase details in the original [Paragraph].\n"
        "* Avoid using pronouns. Be specific when referring to objects, characters or situations.\n"
        "* Do not include too generic sentences (e.g. \"Meme poster is trying to convey a message.\", \"The image is funny.\", .. etc)\n\n"
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
        "    \"The contrast between the doctor’s serious tone and the absurd situation creates comedic irony.\"\n"
        "]\n"
        "```\n\n"
        "--------\n\n"
        "Proceed to break down the following paragraph into a list of atomic sentences.\n"
        "[Paragraph]:\n"
        "{paragraph}\n\n"
        "[Output]:"
    )

    # Stop expressions that should be filtered out from the results.
    STOP_EXPRESSIONS = {
        "The image is ironic.",
        "The image is ironical.",
        "The image is satirical.",
        "The image is funny.",
        "The image is confusing.",
        "The images are ironic.",
        "The images are ironical.",
        "The images are satirical.",
        "The images are funny.",
        "The images are confusing.",
        "Meme poster is trying to convey a message.",
    }

    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=os.environ['GEMINI_KEY'])
        self.gemini = genai.GenerativeModel("gemini-1.5-pro")

    def send_request(self, prompt, temperature, max_tokens):
        """
        Sends a request to Gemini with up to 20 retries.
        """
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

    def decompose(self, paragraph):
        """
        Generates a decomposition of the given paragraph into atomic sentences.
        Returns a list of sentences, filtered against stop expressions.
        """
        prompt = self.FACT_DECOM_PROMPT.format(paragraph=paragraph)
        response = self.send_request(prompt, temperature=0.2, max_tokens=512)
        if not response:
            return []
        try:
            cleaned = response.replace("```python", "").replace("```", "").strip()
            decomposed = eval(cleaned)
        except Exception:
            decomposed = [line.strip() for line in response.split("\n") if line.strip().startswith('"')]
        # Filter out any stop expressions.
        return [fact for fact in decomposed if fact.strip() not in self.STOP_EXPRESSIONS]


def run_decomposition(data, out_file, existing_image_paths, result):
    """
    Iterates over all items in data, decomposes the 'reference' field into atomic sentences,
    and appends a record with the image path, caption, reference, and decomposed reference.
    """
    decomposer = FactDecomposer()
    for item in tqdm.tqdm(data, total=len(data)):
        reference = item.get('reference')
        image_path = item.get('image_file_path')
        image_name = os.path.basename(image_path)

        # Skip if already processed
        if image_name in existing_image_paths:
            print(f"Skipping {image_name}...")
            continue

        print("--------" * 10)
        print("[image_path]:", image_path)
        print("[reference]:", reference)

        # Convert list references to text if necessary.
        if isinstance(reference, list):
            reference_text = " ".join(reference)
        else:
            reference_text = reference

        decomposed_facts = decomposer.decompose(reference_text)
        if not decomposed_facts:
            continue

        print("[Decomposed Facts]:\n", "\n".join([f"\t- {fact}" for fact in decomposed_facts]))

        result.append({
            "image_file_path": image_path,
            "caption": item.get('caption'),
            "reference": reference,
            "decomposed_reference": decomposed_facts
        })

        print("\nFinal result file is saved here:", os.path.abspath(out_file))
        with open(out_file, 'w') as outfile:
            json.dump(result, outfile, indent=4)
    return result


def main(args):
    data_name = args.data_name         # e.g. 'memecap', 'newyorker', 'yesbut'
    model_name = args.model_name       # e.g. 'gpt4o', 'gemini', etc.
    weight = args.weight
    hops = args.hops
    exp_name = args.exp_name
    seed = args.seed

    in_file = f"data/{data_name}/model-output/{model_name}/desc-imp-cand/{exp_name}/model_output_test_t0.8_w{weight}_h{hops}_s0_sd{seed}.json"
    out_dir = "results/fact_decomposed/"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"results/fact_decomposed/seed0/{data_name}_reference_decomposed.json"

    result = []
    if os.path.exists(out_file) and not args.overwrite:
        print(f"Loading existing results from {os.path.basename(out_file)}")
        result = json.load(open(out_file, 'rb'))

    data = json.load(open(in_file, 'rb'))

    existing_image_paths = []
    if result and not args.overwrite:
        print(f"[in_file]: {os.path.basename(in_file)}  [out_file]: {os.path.basename(out_file)}")
        print("Existing records:", len(result))
        for rec in result:
            if "image_file_path" in rec:
                existing_image_paths.append(os.path.basename(rec['image_file_path']))

    result = run_decomposition(data, out_file, existing_image_paths, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='memecap', choices=['memecap', 'newyorker', 'yesbut'], type=str,
                        help='Data directory path')
    parser.add_argument('--model_name', default='gpt4o', type=str, choices=['gpt4o', 'gemini', 'qwen2', 'phi'],
                        help='Model type to generate responses')
    parser.add_argument('--exp_name', default='final', type=str, help='Experiment name')
    parser.add_argument('--hops', default=2, type=int, help='Number of hops')
    parser.add_argument('--weight', default=0.7, type=float, help='Weight parameter')
    parser.add_argument('--seed', default=42, type=int, help='Seed number')
    parser.add_argument('--overwrite', action='store_true', help="Overwrite file if exists")

    args = parser.parse_args()
    main(args)
