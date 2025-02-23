import argparse
from generate_response import GPT4o, Qwen2VL, Phi, Gemini
from transformers import set_seed


def initialize_model(
    data_dir, data_type, model_name, add_cot, add_image_descriptions, add_implications, add_candidate_response, overwrite, weight, num_hops, 
    out_dir_name, split, self_improve, temperature, seed, selection_method
):
    if temperature == -1:
        temperature = None
    if model_name == "gpt4o":
        model = GPT4o.GPT4o(
            model_name=model_name,
            data_dir=data_dir,
            data_type=data_type,
            add_cot=add_cot,
            add_image_descriptions=add_image_descriptions,
            add_implications=add_implications,
            add_candidate_response=add_candidate_response,
            temperature=temperature,
            weight=weight,
            num_hops=num_hops,
            overwrite=overwrite,
            out_dir_name=out_dir_name,
            split=split,
            self_improve=self_improve,
            advanced_model=None,
            seed=seed,
            selection_method=selection_method,
        )
    elif model_name == "qwen2":
        model = Qwen2VL.Qwen2VL(
            model_name=model_name,
            data_dir=data_dir,
            data_type=data_type,
            add_cot=add_cot,
            add_image_descriptions=add_image_descriptions,
            add_implications=add_implications,
            add_candidate_response=add_candidate_response,
            overwrite=overwrite,
            weight=weight,
            num_hops=num_hops,
            out_dir_name=out_dir_name,
            split=split,
            self_improve=self_improve,
            temperature=temperature,
            advanced_model=None,
            seed=seed,
            selection_method=selection_method,
        )
    elif model_name == "phi":
        model = Phi.Phi(
            model_name=model_name,
            data_dir=data_dir,
            data_type=data_type,
            add_cot=add_cot,
            add_image_descriptions=add_image_descriptions,
            add_implications=add_implications,
            add_candidate_response=add_candidate_response,
            overwrite=overwrite,
            weight=weight,
            num_hops=num_hops,
            out_dir_name=out_dir_name,
            split=split,
            self_improve=self_improve,
            temperature=temperature,
            advanced_model=None,
            seed=seed,
            selection_method=selection_method,
        )
    elif model_name == "gemini":
        model = Gemini.Gemini(
            model_name=model_name,
            data_dir=data_dir,
            data_type=data_type,
            add_cot=add_cot,
            add_image_descriptions=add_image_descriptions,
            add_implications=add_implications,
            add_candidate_response=add_candidate_response,
            temperature=temperature,
            weight=weight,
            num_hops=num_hops,
            overwrite=overwrite,
            out_dir_name=out_dir_name,
            split=split,
            self_improve=self_improve,
            advanced_model=None,
            seed=seed,
            selection_method=selection_method,
        )
    else:
        raise Exception(f"Invalid model name -- {model_name}")
    return model

def main(args):
    set_seed(args.seed)
    
    if args.selection_method == "cosine":
        args.weight = 0.0
        print(f"selection method is cosine. Weight updated to {args.weight}")
    
    if args.add_implications:
        args.add_image_descriptions = True
        print(f"add_implications is {args.add_implications}. args.add_image_descriptions updated to {args.add_image_descriptions}")
    
    model = initialize_model(
        data_dir=args.data_dir,
        data_type=args.data_type,
        model_name=args.model_name,
        add_cot=args.add_cot,
        add_image_descriptions=args.add_image_descriptions,
        add_implications=args.add_implications,
        add_candidate_response=args.add_candidate_response,
        overwrite=args.overwrite,
        weight=args.weight,
        temperature=args.temperature,
        num_hops=args.num_hops,
        out_dir_name=args.out_dir_name,
        split=args.split,
        self_improve=args.self_improve,
        seed=args.seed,
        selection_method=args.selection_method,
    )

    model.load_dataset()
    model.load_model()
    model.generate_explanation()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/vflute/', type=str, help='data directory path')
    parser.add_argument('--data_type', default='valid', type=str, choices=['all', 'train', 'valid', 'test'], help='data type to load')
    parser.add_argument('--model_name', default='phi', type=str, choices=['gpt4o', 'gemini', 'qwen2', 'phi'], help='model type to generate responses')
    parser.add_argument('--selection_method', default='', type=str, choices=['', 'random', 'ce', 'cosine'], help='method to use when selecting implications')

    parser.add_argument('--add_cot', action='store_true', help="add chain of thought prompting stlye")
    parser.add_argument('--add_image_descriptions', action='store_true', help="generate image descriptions")
    parser.add_argument('--add_implications', action='store_true', help="generate implications")
    parser.add_argument('--add_candidate_response', action='store_true', help="add candidate response")
    parser.add_argument('--overwrite', action='store_true', help="overwrite file")
    
    parser.add_argument('--weight', default=0.7, type=float, help='weight to control balance between cosine similarity and cross entropy')
    parser.add_argument('--num_hops', default=2, type=int, help='number of hops used in beam search')
    
    parser.add_argument('--out_dir_name', default='', type=str, help='name for model output directory')
    parser.add_argument('--split', default=0, type=int, help='cross validation split number for newyorker dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    
    parser.add_argument('--self_improve', action='store_true', help="self improve")
    parser.add_argument('--temperature', default=0.8, type=float, help='temperature')
    
    args = parser.parse_args()
    main(args)