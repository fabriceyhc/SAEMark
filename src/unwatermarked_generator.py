#encoding=utf-8

import nltk
import torch
import argparse
import numpy as np
from utils import generate
from utils import read_dataset
from transformers import AutoTokenizer
from utils import output_dataset_in_line

def generate_single_sentence_en(input_text, generation_config, port, ip):
    with torch.no_grad():
        times = 0
        while times < 5:
            output_data_seq = generate(input_text, generation_config, ip, port)
            for output_data in output_data_seq:
                output_data = output_data.strip()
                output_data = output_data.replace('\n', ' ')
                if len(output_data) == 0:
                    continue
                sentences = nltk.sent_tokenize(output_data)
                if sentences[0].endswith('.') and len(sentences[0]) > 15:
                    return sentences[0]
            times += 1
        return ''

def generate_single_sentence_zh(input_text, generation_config, port, ip):
    with torch.no_grad():
        times = 0
        while times < 5:
            output_data_seq = generate(input_text, generation_config, ip, port)
            for output_data in output_data_seq:
                output_data = output_data.strip()
                output_data = output_data.replace('\n', ' ')
                if '。' not in output_data:
                    continue
                output_data = output_data.split('。')[0] + '。'
                return output_data
            times += 1
        return ''

def generate_paragraph_text(input_text, generation_config, sent_num, port, lang='en', ip=None):
    output = ''
    for _ in range(sent_num):
        if lang == 'en':
            sentence = generate_single_sentence_en(input_text, generation_config, port, ip)
            input_text += ' ' + sentence
            output = sentence if len(output) == 0 else output + ' ' + sentence
        else:
            sentence = generate_single_sentence_zh(input_text, generation_config, port, ip)
            input_text += sentence
            output += sentence
    return output

def generate_paragraph_code(input_text, generation_config, units, port, ip):
    output_text = ''
    input_text += '```python\n'
    for _ in range(units):
        candidates = generate(input_text, generation_config, ip, port)
        candidates = [(candidate.split('\n')[0] + '\n') for candidate in candidates]
        index = np.random.randint(0, len(candidates), 1)[0]
        if '```' in candidates[index]:
            break
        input_text += candidates[index]
        output_text += candidates[index]
    return output_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified unwatermarked text generator")
    
    parser.add_argument('--dataset', type=str, help="Input dataset path")
    parser.add_argument('--start', type=int, help="Start index in dataset")
    parser.add_argument('--end', type=int, help="End index in dataset")
    parser.add_argument('--base', type=str, help="Path to model for tokenizer (code only)")
    parser.add_argument('--candidates', type=int, help='Number of candidates to sample')
    parser.add_argument('--units', type=int, help="Number of generation units")
    parser.add_argument('--ip', type=str, help='TGI server IP')
    parser.add_argument('--port', type=int, help='TGI server port')
    parser.add_argument('--output_path', type=str, help="Output file path")
    parser.add_argument('--lang', type=str, default='en', help="Language mode: en/zh/code")
    
    args = parser.parse_args()
    data = read_dataset(args.dataset, args.start, args.end)
    
    # Configure generation parameters based on language
    if args.lang == 'code':
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        generation_config = {
            "top_k": 100,
            "top_p": 0.96,
            "do_sample": True,
            "best_of": args.candidates,
            "temperature": 0.8,
            "max_new_tokens": 20,
            "repetition_penalty": 1.2
        }
    else:
        generation_config = {
            "do_sample": True,
            "best_of": args.candidates,
            "temperature": 0.7,
            "max_new_tokens": 30 if args.lang == 'zh' else 20,
        }
        if args.lang == 'zh':
            generation_config["stop_sequences"] = ['。']
    
    for i, item in enumerate(data):
        try:
            if args.lang == 'code':
                messages = [
                    {"role": "system", "content": "You are an expert Python programmer. Your task is to complete the given function in multiple ways, exploring a variety of implementation approaches to demonstrate different techniques and styles. Here is your task:"},
                    {"role": "user", "content": f'{item["prompt"]}'}
                ]
                input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                output = generate_paragraph_code(input_text, generation_config, args.units, args.port, args.ip)
            else:
                output = generate_paragraph_text(item['prompt'], generation_config, args.units, args.port, args.lang, args.ip)
        except Exception as e:
            print(e)
            break
            
        item['unwatermarked'] = output
        output_dataset_in_line(args.output_path, item)
        print(f"Generation {i} complete!")
