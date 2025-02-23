# BottleHumor: Self-Informed Humor Explanation using the Information Bottleneck Principle
[BottleHumor]() is a method inspired by the information bottleneck principle that elicits relevant world knowledge from vision and language models which is iteratively refined for generating an explanation of the humor in an unsupervised manner. Our method can further be adapted in the future for additional tasks that can benefit from eliciting and conditioning on relevant world knowledge and open new research avenues in this direction.


## Installation

* Tested with: Python 3.9, PyTorch 2.4.0, transformers==4.45.2, and sentence-transformers==3.1.0.
* See requirements.txt for more details.


## Data

* We used [MemeCap](https://github.com/eujhwang/meme-cap), [NewYorker Cartoons](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest), and [YesBut](https://huggingface.co/datasets/bansalaman18/yesbut)
* For NewYorker Cartoon dataset, we used test splits of explanation task.
* When running the first time, the code will first download and preprocess the data. It will reuse these files with later runs.

## Model

* For experiments, we used GPT4o (api version: 2024-08-01-preview), [Gemini-1.5-Flash-8B](https://ai.google.dev/gemini-api/docs/pricing), [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [Phi-3.5-Vision-Instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct).
* For evaluation, we used [Gemini-1.5-Flash](https://ai.google.dev/gemini-api/docs/pricing).

## Example Command

* Zero-shot baseline: <br />
`python run.py --data_dir=data/memecap/ --data_type=test --model_name=gpt4o --split=0 --num_hops=2 --out_dir_name='final' --temperature=0.8 --weight=0.7 --seed=0`

* Chain-of-Thought (CoT) baseline: <br />
`python run.py --data_dir=data/memecap/ --data_type=test --model_name=gpt4o --add_cot --split=0 --num_hops=2 --out_dir_name='final' --temperature=0.8 --weight=0.7 --seed=0`

* Ours: <br />
`python run.py --data_dir=data/memecap/ --data_type=test --model_name=gpt4o --add_image_descriptions --add_implications --add_candidate_response --split=0 --num_hops=2 --out_dir_name='final' --temperature=0.8 --weight=0.7 --seed=0`


## Example Command for Evaluation

* Precision <br />
`python llm_precision_evaluator.py --data_name=memecap --model_name=gpt4o --weight=0.7 --exp_name=[out_dir_name] --seed=0`

* Recall <br />
`python llm_recall_evaluator.py --data_name=memecap --model_name=gpt4o --weight=0.7 --exp_name=[out_dir_name] --seed=0`

## Paper

If the code is helpful for your project, please cite [our paper]() (Bibtex below).
```
TBD
```

## Note

SentenceSHAP is adapted from TokenSHAP code: https://github.com/ronigold/TokenSHAP