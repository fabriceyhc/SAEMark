# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ================================================
# text_editor.py
# Description: Edit text using various techniques
# ================================================

import nltk
import torch
import random
from nltk.corpus import wordnet
from transformers import BertTokenizer, BertForMaskedLM

class TextEditor:
    """Base class for text editing."""

    def __init__(self) -> None:
        pass

    def edit(self, text: str, reference=None):
        return text


class WordDeletion(TextEditor):
    """Delete words randomly from the text."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the word deletion editor.

            Parameters:
                ratio (float): The ratio of words to delete.
        """
        self.ratio = ratio

    def edit(self, text: str, reference=None):
        """Delete words randomly from the text."""

        # Handle empty string input
        if not text:  
            return text

        # Split the text into words and randomly delete each word based on the ratio
        word_list = text.split()
        black_list = ['.' , ',', '!', '?', ':', ';']
        def check(word):
            for item in black_list:
                if item in word:
                    return True
            return False
        edited_words = [word for word in word_list if random.random() >= self.ratio or check(word)]

        # Join the words back into a single string
        deleted_text = ' '.join(edited_words)

        return deleted_text


class SynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the synonym substitution editor.

            Parameters:
                ratio (float): The ratio of words to replace.
        """
        self.ratio = ratio
        # Ensure wordnet data is available
        nltk.download('wordnet')

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet."""
        words = text.split()
        black_list = ['.' , ',', '!', '?', ':', ';']
        def check(word):
            for item in black_list:
                if item in word:
                    return True
            return False
        num_words = len(words)
        
        # Dictionary to cache synonyms for words
        word_synonyms = {}

        # First pass: Identify replaceable words and cache their synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if check(word):
                continue
            if word not in word_synonyms:
                synonyms = [syn for syn in wordnet.synsets(word) if len(syn.lemmas()) > 1]
                word_synonyms[word] = synonyms
            if word_synonyms[word]:
                replaceable_indices.append(i)

        # Calculate the number of words to replace
        num_to_replace = min(int(self.ratio * num_words), len(replaceable_indices))

        # Randomly select words to replace
        if num_to_replace > 0:
            indices_to_replace = random.sample(replaceable_indices, num_to_replace)
        
            # Perform replacement
            for i in indices_to_replace:
                synonyms = word_synonyms[words[i]]
                chosen_syn = random.choice(synonyms)
                new_word = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
                words[i] = new_word

        # Join the words back into a single string
        replaced_text = ' '.join(words)

        return replaced_text


class ContextAwareSynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet based on the context."""

    def __init__(self, ratio: float, tokenizer: BertTokenizer, model: BertForMaskedLM, device='cuda') -> None:
        """
        Initialize the context-aware synonym substitution editor.

        Parameters:
            ratio (float): The ratio of words to replace.
            tokenizer (BertTokenizer): Tokenizer for BERT model.
            model (BertForMaskedLM): BERT model for masked language modeling.
            device (str): Device to run the model (e.g., 'cuda', 'cpu').
        """
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        nltk.download('wordnet')
    
    def _get_synonyms_from_wordnet(self, word: str):
        """ Return a list of synonyms for the given word using WordNet. """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet based on the context."""
        words = text.split()
        black_list = ['.' , ',', '!', '?', ':', ';']
        def check(word):
            for item in black_list:
                if item in word:
                    return True
            return False
        num_words = len(words)
        replaceable_indices = []

        for i, word in enumerate(words):
            if check(word):
                continue
            if self._get_synonyms_from_wordnet(word):
                replaceable_indices.append(i)

        num_to_replace = int(min(self.ratio, len(replaceable_indices) / num_words) * num_words)
        indices_to_replace = random.sample(replaceable_indices, num_to_replace)

        real_replace = 0

        for i in indices_to_replace:
            # Create a sentence with a [MASK] token
            masked_sentence = words[:i] + ['[MASK]'] + words[i+1:]
            masked_text = " ".join(masked_sentence)
            
            # Use BERT to predict the token for [MASK]
            inputs = self.tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
            mask_position = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0].item()

            with torch.no_grad():
                outputs = self.model(**inputs)

            predictions = outputs.logits[0, mask_position]
            predicted_indices = torch.argsort(predictions, descending=True)
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices[0:1])
            words[i] = predicted_tokens[0]
            real_replace += 1
        
        replaced_text = ' '.join(words)

        return replaced_text