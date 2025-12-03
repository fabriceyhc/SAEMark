#encoding:utf-8

import nltk
import torch
from utils import generate
nltk.download('punkt_tab')
from fcs import calc_ratio
from utils import generate_precise_number

def generate_single_sentence_en(
        input_data, 
        prev_max_ind=-1,
        num_return_sequences=20,
        public_key=None,
        device='cuda:7',
        interval=None,
        ip=None,
        port=None,
        feature_mask=None,
        generation_config=None,
        judge_model=None,
        judge_tokenizer=None,
        path_to_params=None
):
    if prev_max_ind != -1:
        precise_number = generate_precise_number(public_key, str(prev_max_ind), interval[0], interval[1])
    else:
        precise_number = generate_precise_number(public_key, '', interval[0], interval[1])
    
    with torch.no_grad():
        candidates = list()
        times = 0
        while num_return_sequences > len(candidates) and times < 3:
            # if there is no enough candidates
            output_data_seq = generate(input_data, generation_config, ip, port)
            # analyze the outputs and get the candidates
            for output_data in output_data_seq:
                output_data = output_data.strip()
                output_data = output_data.replace('\n', ' ')
                # if the output_data is empty
                if len(output_data) == 0:
                    continue
                sentences = nltk.sent_tokenize(output_data)
                if sentences[0].endswith('.'):
                    # if the sentence is not exists in current sentences
                    if sentences[0] not in candidates:
                        candidates.append(sentences[0])
            times += 1
        # remove the residual part
        if len(candidates) > num_return_sequences:
            candidates = candidates[:num_return_sequences]
        ans = calc_ratio(judge_model, judge_tokenizer, candidates, path_to_params, device, feature_mask)
        # the 'rank' array is for sentence rank
        rank = list()
        for i, item in enumerate(ans):
            rank.append((candidates[i], item[0], item[1]))
        # sort the candidates by the ratio
        rank.sort(key=lambda x: abs(x[2] - precise_number))
        return round(precise_number, 4), rank[0][0], rank[0][1], rank[0][2]


def generate_paragraph_en(
    input_text, 
    ret_seq, 
    sent_num, 
    device, 
    interval, 
    public_key, 
    ip,
    port,
    feature_mask,
    generation_config,
    judge_model,
    judge_tokenizer,
    path_to_params
):
    prev_max_ind = -1
    precise_numbers = []
    actual_numbers = []

    output_text = ''
    for _ in range(sent_num):
        precise_number, sentence, max_ind, res = \
            generate_single_sentence_en(
                input_data=input_text,
                prev_max_ind=prev_max_ind,
                num_return_sequences=ret_seq,
                device=device,
                interval=interval,
                public_key=public_key,
                ip=ip,
                port=port,
                feature_mask=feature_mask,
                generation_config=generation_config,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                path_to_params=path_to_params
            )
        prev_max_ind = max_ind
        precise_numbers.append(precise_number)
        actual_numbers.append(res)
        input_text += ' ' + sentence
        # calculate the output text
        if len(output_text) == 0:    output_text = sentence
        else: output_text += ' ' + sentence
    actual_min = min(actual_numbers)
    actual_max = max(actual_numbers)
    precise_min = min(precise_numbers)
    precise_max = max(precise_numbers)
    cross_min = max(actual_min, precise_min)
    cross_max = min(actual_max, precise_max)
    if precise_max - precise_min == 0:
        return output_text, 0, 0
    # calculate the rank
    rank1 = (actual_max - actual_min) / (precise_max - precise_min)
    rank2 = (cross_max - cross_min) / (precise_max - precise_min)
    return output_text, rank1, rank2