#encoding=utf-8
import argparse
from utils import read_dataset
from utils import output_dataset_in_line
from transformers import BertTokenizer, BertForMaskedLM
from attack_operations.operations import WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text attack parameters")
    
    parser.add_argument('--input_path', type=str, help='Path to input data file')
    parser.add_argument('--watermarked', type=str, help='Whether the data is watermarked.')
    parser.add_argument('--ratio', type=float, help='Ratio of words to modify')
    parser.add_argument('--output_path', type=str, help='Path to output data file')
    parser.add_argument('--attack_type', type=str, choices=['deletion', 'syno_sub', 'context_sub'],
                      help='Type of attack to perform (deletion, synonym substitution, or context-aware substitution)')
    parser.add_argument('--bert_path', type=str, help='The path of bert model for context aware substitution.')
    parser.add_argument('--device', type=str, help='Device to use for context-aware substitution (e.g., cuda:0)',
                      default='cpu')
    
    args = parser.parse_args()
    
    # Load dataset
    data = read_dataset(args.input_path, start=0, end=500)
    
    # Initialize attack based on type
    if args.attack_type == 'deletion':
        attack = WordDeletion(ratio=args.ratio)
    elif args.attack_type == 'syno_sub':
        attack = SynonymSubstitution(ratio=args.ratio)
    else:  # context_sub
        attack = ContextAwareSynonymSubstitution(
            ratio=args.ratio,
            tokenizer=BertTokenizer.from_pretrained(args.bert_path),
            model=BertForMaskedLM.from_pretrained(args.bert_path).to(args.device),
            device=args.device
        )
    
    # Apply attack to each item
    for i, item in enumerate(data):
        if not args.watermarked:
            item['unwatermarked'] = attack.edit(item['unwatermarked'])
        else:
            item['watermarked'] = attack.edit(item['watermarked'])
        output_dataset_in_line(args.output_path, item)
        print(f'{args.attack_type} attack on text {i} is done.')
