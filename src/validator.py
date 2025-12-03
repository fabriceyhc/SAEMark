#encoding:utf-8

# this script is used to validate the sentences

import argparse
from utils import read_dataset
from utils import get_public_key
from utils import generate_low_high_edges
from huggingface_hub import hf_hub_download
from utils import output_dataset_in_line
from transformers import AutoModelForCausalLM, AutoTokenizer

# set the gemma2 model to judge
judge_model = None
judge_tokenizer = None
path_to_params = None

def init(device, anchor):

    # Load the judgement model -- gemma-2-2b
    global judge_model
    judge_model = AutoModelForCausalLM.from_pretrained(
        anchor,
        device_map=device
    )

    # Load the judgement tokenizer
    global judge_tokenizer
    judge_tokenizer = AutoTokenizer.from_pretrained(anchor)

    # Load the judgement model parameters
    global path_to_params
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-res", 
        filename="layer_20/width_16k/average_l0_71/params.npz",
        force_download=False,
    )
    
def generate_range(private_keys, mu=0.1271, sigma=0.0204):
    intervals = list()
    for private_key in private_keys:
        low_edge, high_edge = generate_low_high_edges(private_key=private_key, mu=mu, sigma=sigma)
        intervals.append([[low_edge, mu - sigma * 0.5], [mu + sigma * 0.5, high_edge]])
    return intervals

if __name__ == '__main__':
    # add the require imports
    from fcs import get_mask
    from validator_operations.roc_score import validate_roc_en
    from validator_operations.roc_score import validate_roc_zh
    from validator_operations.multi_users import validate_multi_en
    from validator_operations.multi_users import validate_multi_zh
    from validator_operations.roc_score import validate_roc_code
    from validator_operations.multi_users import validate_multi_code
    
    parser = argparse.ArgumentParser(description="The arguments of the generator.")

    parser.add_argument('--input_path', type=str, help='The input data of the validator.')
    parser.add_argument('--watermarked', type=bool, help='Whether the data is watermarked.')
    parser.add_argument('--start', type=int, help="The start to handle in the input data.")
    parser.add_argument('--end', type=int, help="The end to handle in the input data.")
    parser.add_argument('--mask', type=bool, help='The mask of the feature.')
    parser.add_argument('--anchor', type=str, help='The anchor model to generate.')
    parser.add_argument('--device', type=str, help='The device to calculate on.')
    parser.add_argument('--mu', type=float, help='The mu for generate.')
    parser.add_argument('--sigma', type=float, help='The sigma for generate.')
    parser.add_argument('--rmin', type=float, help='The rmin for range similarity.')
    parser.add_argument('--rmax', type=float, help='The rmax for range similarity.')
    parser.add_argument('--omin', type=float, help='The omin for overlap ratio.')
    parser.add_argument('--lang', type=str, help='The language for validation.')
    parser.add_argument('--task', type=str, help='The task to validate. For example, \'multi\' or \'roc\'')
    parser.add_argument('--users', type=int, help='The user number for multi user check.', default=1)
    parser.add_argument('--output_path', type=str, help='The output data of the validator.')

    args = parser.parse_args()
    
    # set the μ and σ for generate
    μ = args.mu
    σ = args.sigma

    device = args.device

    init(device, args.anchor)

    # set the keys
    private_keys = [str(i) for i in range(1, args.users + 1)]

    intervals = generate_range(private_keys, μ, σ)
    if args.mask:
        # get the restricted
        content = read_dataset('dataset/bg_freq_feat_mask/original.json', start=0, end=3)
        for elem in content:
            if elem['lang'] == args.lang:
                restricted = elem['list']
    else:
        restricted = []

    # get the feature mask
    feature_mask = get_mask(device, restricted)
    
    public_keys = list()
    for private_key in private_keys:
        public_key = get_public_key(private_key)
        public_keys.append(public_key)

    data = read_dataset(args.input_path, args.start, args.end)
    
    if args.task == 'multi':
        
        for i, item in enumerate(data):
            # if the data is unwatermarked data
            if not args.watermarked:
                line = item['unwatermarked']
            else:
                line = item['watermarked']
            if args.lang == 'zh':
                detect = validate_multi_zh(
                    sentence=line, 
                    public_keys=public_keys, 
                    device=device, 
                    intervals=intervals,  
                    feature_mask=feature_mask,
                    judge_model=judge_model,
                    judge_tokenizer=judge_tokenizer,\
                    path_to_params=path_to_params,
                    rmin=args.rmin,
                    rmax=args.rmax,
                    omin=args.omin
                )
                item['detection_key'] = str(detect)
                output_dataset_in_line(args.output_path, item)
                print(f"Validation {i} complete!")
            if args.lang == 'en':
                detect = validate_multi_en(
                    sentence=line, 
                    public_keys=public_keys, 
                    device=device, 
                    intervals=intervals, 
                    feature_mask=feature_mask,
                    judge_model=judge_model,
                    judge_tokenizer=judge_tokenizer,\
                    path_to_params=path_to_params,
                    rmin=args.rmin,
                    rmax=args.rmax,
                    omin=args.omin
                )
                item['detection_key'] = str(detect)
                output_dataset_in_line(args.output_path, item)
                print(f"Validation {i} complete!")
            if args.lang == 'code':
                detect = validate_multi_code(
                    sentence=line, 
                    public_keys=public_keys, 
                    device=device, 
                    intervals=intervals, 
                    feature_mask=feature_mask,
                    judge_model=judge_model,
                    judge_tokenizer=judge_tokenizer,\
                    path_to_params=path_to_params,
                    rmin=args.rmin,
                    rmax=args.rmax,
                    omin=args.omin
                )
                item['detection_key'] = str(detect)
                output_dataset_in_line(args.output_path, item)
                print(f"Validation {i} complete!")
                
    elif args.task == 'roc':
        
        for i, item in enumerate(data):
            res = dict()
            # if the data is unwatermarked data
            if 'uwm' in args.input_path:
                line = item['unwatermarked']
                res['label'] = 0
            else:
                line = item['watermarked']
                res['label'] = 1
            if args.lang == 'zh':
                score = validate_roc_zh(
                    sentence=line, 
                    public_keys=public_keys, 
                    device=device, 
                    intervals=intervals,  
                    feature_mask=feature_mask,
                    judge_model=judge_model,
                    judge_tokenizer=judge_tokenizer,\
                    path_to_params=path_to_params
                )
                res['score'] = score
                output_dataset_in_line(args.output_path, res)
                print(f"Validation {i} complete!")
            if args.lang == 'en':
                score = validate_roc_en(
                    sentence=line, 
                    public_keys=public_keys, 
                    device=device, 
                    intervals=intervals, 
                    feature_mask=feature_mask,
                    judge_model=judge_model,
                    judge_tokenizer=judge_tokenizer,\
                    path_to_params=path_to_params
                )
                res['score'] = score
                output_dataset_in_line(args.output_path, res)
                print(f"Validation {i} complete!")
            if args.lang == 'code':
                score = validate_roc_code(
                    sentence=line, 
                    public_keys=public_keys, 
                    device=device, 
                    intervals=intervals, 
                    feature_mask=feature_mask,
                    judge_model=judge_model,
                    judge_tokenizer=judge_tokenizer,\
                    path_to_params=path_to_params
                )
                res['score'] = score
                output_dataset_in_line(args.output_path, res)
                print(f"Validation {i} complete!")