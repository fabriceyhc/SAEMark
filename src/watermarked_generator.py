#encoding:utf-8

import argparse
from utils import read_dataset
from utils import get_public_key
from huggingface_hub import hf_hub_download
from utils import output_dataset_in_line
from utils import generate_low_high_edges
from transformers import AutoModelForCausalLM, AutoTokenizer

# the gemma2 model to judge
judge_model = None
judge_tokenizer = None
path_to_params = None

# the generation config to tgi
generation_config = None

def init(device, ret_seq, lang, anchor):

    # Load the SAE Model --gemma-2-2b
    global judge_model
    judge_model = AutoModelForCausalLM.from_pretrained(
        anchor,
        device_map=device
    )
    # Load the judgement tokenizer
    global judge_tokenizer
    judge_tokenizer = AutoTokenizer.from_pretrained(anchor)
    # Load the SAE Model parameters
    global path_to_params
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-res", 
        filename="layer_20/width_16k/average_l0_71/params.npz",
        force_download=False,
    )
    global generation_config
    if lang == 'en':
        generation_config = {
            "do_sample": True,
            "best_of": ret_seq,
            "temperature": 0.7,
            "max_new_tokens": 50,
        }
    elif lang == 'zh':
        generation_config = {
            "do_sample": True,
            "best_of": ret_seq,
            "temperature": 0.7,
            "max_new_tokens": 30,
            "stop_sequences": ['。']
        }
    elif lang == 'code':
        generation_config = {
            "top_k": 100,
            "top_p": 0.96,
            "do_sample": True,
            "best_of": ret_seq,
            "temperature": 0.8,
            "max_new_tokens": 20,
            "repetition_penalty": 1.2
        }

if __name__ == "__main__":
    # get the imports for generations
    from fcs import get_mask
    from generator_operations.en import generate_paragraph_en
    from generator_operations.zh import generate_paragraph_zh
    from generator_operations.code import generate_paragraph_code

    parser = argparse.ArgumentParser(description="The arguments of the generator.")

    parser.add_argument('--dataset', type=str, help='The input data of the generator.')
    parser.add_argument('--start', type=int, help="The start to handle in dataset.")
    parser.add_argument('--end', type=int, help="The end to handle in dataset.")
    parser.add_argument('--mask', type=bool, help='The mask of the feature.')
    parser.add_argument('--base', type=str, help='The base model to generate.')
    parser.add_argument('--anchor', type=str, help='The anchor model for fcs.')
    parser.add_argument('--device', type=str, help='The device to calculate on.')
    parser.add_argument('--mu', type=float, help='The mu for generate.')
    parser.add_argument('--sigma', type=float, help='The sigma for generate.')
    parser.add_argument('--rmin', type=float, help='The rmin for range similarity.')
    parser.add_argument('--rmax', type=float, help='The rmax for range similarity.')
    parser.add_argument('--omin', type=float, help='The omin for overlap ratio.')
    parser.add_argument('--lang', type=str, help='The language for judging, for example \'en\', \'code\' and \'zh\'')
    parser.add_argument('--candidates', type=int, help='The sampling num.')
    parser.add_argument('--units', type=int, help='The iterations of generation.')
    parser.add_argument('--attempts', type=int, help='The repeat time of selection.')
    parser.add_argument('--ip', type=str, help='The ip of the tgi.')
    parser.add_argument('--port', type=int, help='The port of tgi.')
    parser.add_argument('--output_path', type=str, help='The output data of the generator.')

    args = parser.parse_args()
    
    # set the μ and σ for generate
    μ = args.mu
    σ = args.sigma

    device = args.device
    
    if args.mask:
        # get the restricted list
        content = read_dataset('dataset/bg_freq_feat_mask/original.json', start=0, end=3)
        for elem in content:
            if elem['lang'] == args.lang:
                restricted = elem['list']
    else:
        restricted = []
            
    # get the feature mask
    feature_mask = get_mask(device, restricted)

    # init the models
    init(device, args.candidates, args.lang, args.anchor)

    data = read_dataset(args.dataset, args.start, args.end)
    
    # init the tokenizer for code generator
    if args.lang == 'code':
        tokenizer = AutoTokenizer.from_pretrained(args.base)

    for i, item in enumerate(data):
        private_key = item['target_key']
        # calc the low range and high range
        low_edge, high_edge = generate_low_high_edges(private_key=private_key, mu=μ, sigma=σ)
        interval = [[low_edge, μ - σ * 0.5], [μ + σ * 0.5, high_edge]]
        
        # use the private key to generate the public key
        public_key = get_public_key(private_key=private_key)
        # get the prompt
        input_text = item['prompt']
        repeat_time = 0
        while repeat_time < args.attempts:
            try:
                if args.lang == 'zh':
                    output, rank1, rank2 = generate_paragraph_zh(
                        input_text=input_text,
                        ret_seq=args.candidates,
                        sent_num=args.units,
                        device=device,
                        interval=interval,
                        public_key=public_key,
                        ip=args.ip,
                        port=args.port,
                        feature_mask=feature_mask,
                        generation_config=generation_config,
                        judge_model=judge_model,
                        judge_tokenizer=judge_tokenizer,
                        path_to_params=path_to_params
                    )
                    if args.rmin < rank1 < args.rmax and rank2 >= args.omin:
                        break
                    repeat_time += 1
                elif args.lang == 'en':
                    output, rank1, rank2 = generate_paragraph_en(
                        input_text=input_text,
                        ret_seq=args.candidates,
                        sent_num=args.units,
                        device=device,
                        interval=interval,
                        public_key=public_key,
                        ip=args.ip,
                        port=args.port,
                        feature_mask=feature_mask,
                        generation_config=generation_config,
                        judge_model=judge_model,
                        judge_tokenizer=judge_tokenizer,
                        path_to_params=path_to_params
                    )
                    if args.rmin < rank1 < args.rmax and rank2 >= args.omin:
                        break
                    repeat_time += 1
                elif args.lang == 'code':
                    messages = [
                        {
                            "role": "system", 
                            "content": "You are an expert Python programmer. \
                                Your task is to complete the given function in multiple ways, \
                                    exploring a variety of implementation approaches to demonstrate \
                                        different techniques and styles. Here is your task:"
                        },
                        {
                            "role": "user", 
                            "content": f'{input_text}'
                        }
                    ]
                    input_text = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    input_text += '```python\n'
                    output, rank1, rank2 = generate_paragraph_code(
                        input_text=input_text,
                        ret_seq=args.candidates,
                        sent_num=args.units,
                        device=device,
                        interval=interval,
                        public_key=public_key,
                        ip=args.ip,
                        port=args.port,
                        feature_mask=feature_mask,
                        generation_config=generation_config,
                        judge_model=judge_model,
                        judge_tokenizer=judge_tokenizer,
                        path_to_params=path_to_params
                    )
                    if args.rmin < rank1 < args.rmax and rank2 >= args.omin:
                        break
                    repeat_time += 1
            except Exception as e:
                continue
        item['watermarked'] = output
        output_dataset_in_line(args.output_path, item)
        print(f"Generation {i} complete!")