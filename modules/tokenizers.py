import json
import re
from collections import Counter
from typing import List, Optional
import torch
import os
from transformers import AutoTokenizer



class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        self.vocab_path = args.vocab_path # Assumes args has vocab_path

        # This conditional logic correctly selects the cleaning function based on the dataset.
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        elif self.dataset_name == 'wsi_report':
            self.clean_report = self.clean_report_pathology
        else:
            self.clean_report = self.clean_report_mimic_cxr

        # --- Load or Create Vocabulary ---
        if os.path.exists(self.vocab_path):
            print(f"INFO: Loading vocabulary from: {self.vocab_path}")
            self.token2idx, self.idx2token = self.load_vocabulary(self.vocab_path)
        else:
            print(f"INFO: Vocabulary not found. Building from scratch from: {self.ann_path}")
            self.ann = json.loads(open(self.ann_path, 'r').read())
            self.token2idx, self.idx2token = self.create_vocabulary()
            # Save the newly created vocabulary for future use
            self.save_vocabulary(self.vocab_path)

    # NEW METHOD TO SAVE VOCABULARY
    def save_vocabulary(self, vocab_path):
        """Saves the current vocabulary to a JSON file."""
        print(f"INFO: Saving vocabulary to: {vocab_path}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        try:
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump({'token2idx': self.token2idx, 'idx2token': self.idx2token}, f, indent=4)
        except Exception as e:
            print(f"ERROR: Could not save vocabulary to {vocab_path}: {e}")
            raise

    # NEW METHOD TO LOAD VOCABULARY
    def load_vocabulary(self, vocab_path):
        """Loads the vocabulary from a JSON file."""
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # JSON saves all keys as strings, so convert idx2token keys back to int
            idx2token = {int(k): v for k, v in data['idx2token'].items()}
            return data['token2idx'], idx2token
        except Exception as e:
            print(f"ERROR: Could not load vocabulary from {vocab_path}: {e}")
            raise

    def create_vocabulary(self):
        # (This method remains the same as before)
        total_tokens = []
        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
            
        print(f"INFO: Vocabulary built for {self.dataset_name}. Size: {len(token2idx)}")
        return token2idx, idx2token

    # NEW, IMPROVED METHOD FOR PATHOLOGY REPORTS
    def clean_report_pathology(self, report: str) -> str:
        """
        A conservative cleaning function optimized for modern biomedical LLMs.
        It preserves case, structure, and crucial punctuation.
        """
        # 1. Normalize newlines and strip whitespace. Case is preserved.
        report = report.replace('\n', ' ').strip()
        report = re.sub(r'\s{2,}', ' ', report)

        # 2. Add spaces around key punctuation to ensure they are treated as separate tokens.
        report = re.sub(r'([():;,])', r' \1 ', report)

        # 3. Handle periods carefully to separate from words without affecting decimals.
        report = re.sub(r'(?<!\d)\.(?!\d)', ' . ', report)
        report = re.sub(r'(?<=\d)\.(?!\d)', ' . ', report)
        report = re.sub(r'(?<!\d)\.(?=\d)', ' . ', report)

        # 4. Remove a restricted set of characters. Underscores are replaced.
        report = report.replace('_', ' ')
        report = re.sub(r'[#*?^&\[\]{}]', '', report)
        
        # 5. Final space cleanup.
        report = re.sub(r'\s{2,}', ' ', report).strip()
        return report

    # KEPT YOUR ORIGINAL METHODS FOR OTHER DATASETS
    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        # Use 0 as the special start/end token ID as in your original code
        ids = [0] + [self.get_id_by_token(token) for token in tokens] + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                token = self.get_token_by_id(idx)
                # Improved spacing logic
                if token in {'.', ',', ';', ':', ')'} or txt.endswith('('):
                    txt = txt.rstrip() + token
                else:
                    if i > 0:
                        txt += ' '
                    txt += token
            else:
                # Stop decoding at the special 0 token
                break
        return txt.strip()

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
    
# class Tokenizer(object):
#     def __init__(self, args):
#         self.ann_path = args.ann_path
#         self.threshold = args.threshold
#         self.dataset_name = args.dataset_name
#         if self.dataset_name == 'iu_xray':
#             self.clean_report = self.clean_report_iu_xray
#         elif self.dataset_name == 'wsi_report':
#             self.clean_report = self.clean_report_pathology
#         else:
#             self.clean_report = self.clean_report_mimic_cxr
#         self.ann = json.loads(open(self.ann_path, 'r').read())
#         self.token2idx, self.idx2token = self.create_vocabulary()

#     def create_vocabulary(self):
#         total_tokens = []

#         for example in self.ann['train']:
#             tokens = self.clean_report(example['report']).split()
#             for token in tokens:
#                 total_tokens.append(token)

#         counter = Counter(total_tokens)
#         vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
#         vocab.sort()
#         token2idx, idx2token = {}, {}
#         for idx, token in enumerate(vocab):
#             token2idx[token] = idx + 1
#             idx2token[idx + 1] = token
#         return token2idx, idx2token

#     def clean_report_iu_xray(self, report):
#         report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
#             .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
#             .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#             .strip().lower().split('. ')
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
#                                         replace('\\', '').replace("'", '').strip().lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report

#     def clean_report_mimic_cxr(self, report):
#         report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
#             .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
#             .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
#             .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
#             .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
#             .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
#             .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#             .strip().lower().split('. ')
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
#                                         .replace('\\', '').replace("'", '').strip().lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report
    
#     # def clean_report_pathology(self, report):
#     #     report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
#     #         .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
#     #         .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
#     #         .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
#     #         .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
#     #         .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
#     #         .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#     #         .strip().lower().split('. ')
#     #     sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
#     #                                     .replace('\\', '').replace("'", '').strip().lower())
#     #     tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#     #     report = ' . '.join(tokens) + ' .'
#     #     return report
    
#     def clean_report_pathology(self, report: str) -> str:
#         """
#         This function cleans a pathology report using a conservative approach
#         optimized for modern biomedical LLMs.

#         It replaces the old logic entirely with the superior method from
#         clean_report_brca, which preserves case, structure, and crucial punctuation.
#         """
#         # 1. Normalize newlines and strip leading/trailing whitespace. Case is preserved.
#         report = report.replace('\n', ' ').strip()

#         # 2. Add spaces around key punctuation to ensure they are treated as separate tokens.
#         # This prevents them from being merged with words.
#         report = re.sub(r'([():+,;])', r' \1 ', report)

#         # 3. Carefully add spaces around periods to handle sentence ends and numbered lists
#         # without breaking decimal numbers.
#         report = re.sub(r'(?<=\d)\.(?!\d)', r' . ', report)  # Period after a digit but not followed by a digit
#         report = re.sub(r'(?<!\d)\.(?!\d)', r' . ', report) # Period not surrounded by digits
#         report = re.sub(r'(?<!\d)\.(?=\d)', r' . ', report)  # Period before a digit

#         # 4. Remove only a very restricted set of characters that are highly likely to be noise.
#         report = re.sub(r'[#*?^&_\[\]{}]', '', report)

#         # 5. Collapse multiple spaces into one and perform a final strip.
#         report = re.sub(r'\s{2,}', ' ', report).strip()

#         return report


#     def get_token_by_id(self, id):
#         return self.idx2token[id]

#     def get_id_by_token(self, token):
#         if token not in self.token2idx:
#             return self.token2idx['<unk>']
#         return self.token2idx[token]

#     def get_vocab_size(self):
#         return len(self.token2idx)

#     def __call__(self, report):
#         tokens = self.clean_report(report).split()
#         ids = []
#         for token in tokens:
#             ids.append(self.get_id_by_token(token))
#         ids = [0] + ids + [0]
#         return ids

#     def decode(self, ids):
#         txt = ''
#         for i, idx in enumerate(ids):
#             if idx > 0:
#                 if i >= 1:
#                     txt += ' '
#                 txt += self.idx2token[idx]
#             else:
#                 break
#         return txt

#     def decode_batch(self, ids_batch):
#         out = []
#         for ids in ids_batch:
#             out.append(self.decode(ids))
#         return out
    
class ModernTokenizer(object):
    """A wrapper class that can use different modern tokenizers while maintaining compatibility
    with your existing codebase."""
    
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        
        # Option 1: Medical-specific tokenizer (BioClinicalBERT)
        # self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
        # Option 2: General purpose tokenizer (RoBERTa)
        # self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        # Option 3: Domain-adapted tokenizer (PubMedBERT)
        self.tokenizer = AutoTokenizer.from_pretrained("aaditya/Llama3-OpenBioLLM-8B")
        
        # Add special tokens specific to medical reports if needed
        special_tokens = {
            "additional_special_tokens": [
                "<impression>", "</impression>",
                "<findings>", "</findings>",
                "<comparison>", "</comparison>",
                "<indication>", "</indication>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load your existing vocabulary for compatibility
        self.ann = json.loads(open(args.ann_path, 'r').read())
        
        # Map special tokens
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        
        # Create reverse mapping
        self.token2idx = self.tokenizer.get_vocab()
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        
    def __call__(self, report: str) -> List[int]:
        """Convert report text to token ids."""
        # Use the tokenizer's built-in encoding
        encoding = self.tokenizer.encode(
            report,
            add_special_tokens=True,
            max_length=self.args.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding[0].tolist()  # Convert tensor to list
    
    def decode(self, ids: List[int]) -> str:
        """Convert token ids back to text."""
        # Use the tokenizer's built-in decoding
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return self.post_process_medical_text(text)

    def post_process_medical_text(self, text: str) -> str:
        """Apply domain-specific post-processing to medical reports."""
        # Capitalize anatomical terms
        anatomical_terms = {'chest', 'heart', 'lungs', 'pleural', 'mediastinum'}
        for term in anatomical_terms:
            text = text.replace(f' {term} ', f' {term.capitalize()} ')
        
        # Fix common medical abbreviations
        abbreviations = {
            'pa': 'PA',
            'ap': 'AP',
            'lat': 'LAT',
            'iv': 'IV',
            'ct': 'CT'
        }
        for abbr, replacement in abbreviations.items():
            text = text.replace(f' {abbr} ', f' {replacement} ')
        
        # Fix measurements spacing - using a safer regex pattern
        text = re.sub(r'(\d+)(mm|cm|inches)', r'\1 \2', text)
        
        # Proper sentence capitalization
        sentences = text.split('. ')
        sentences = [s.capitalize() for s in sentences if s]
        text = '. '.join(sentences)
        
        return text.strip()
    
    def decode_batch(self, ids_batch: List[List[int]]) -> List[str]:
        """Decode a batch of token ids to texts."""
        return [self.decode(ids) for ids in ids_batch]
    
    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.token2idx)
    
    def get_token_by_id(self, id: int) -> str:
        """Get token string from token id."""
        return self.idx2token.get(id, self.unk_token)
    
    def get_id_by_token(self, token: str) -> int:
        """Get token id from token string."""
        return self.token2idx.get(token, self.token2idx[self.unk_token])
        
    def batch_encode(self, reports: List[str], 
                    max_length: Optional[int] = None) -> torch.Tensor:
        """Encode a batch of reports to token ids."""
        if max_length is None:
            max_length = self.args.max_seq_length
            
        encodings = self.tokenizer.batch_encode_plus(
            reports,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encodings['input_ids']
    

class MedicalReportTokenizer(object):
    def __init__(self, args):
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        # Get vocab_path from args, default to None if not present
        self.vocab_path = getattr(args, 'vocab_path', None) # Safer way to get attribute

        self.vocabulary_was_built = False # Flag to indicate if vocab was built or loaded

        # Special tokens
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        
        # Add medical domain vocabulary
        self.domain_vocab = set([
            # Common anatomical terms
            'granuloma', 'parametria', 'lymph', 'nodes', 'cervix', 'uterus',
            'ovary', 'ovaries', 'fallopian', 'tubes', 'endometrium', 'myometrium',
            'serosa', 'parametrium', 'vagina', 'vulva',
            
            # Common pathological terms
            'carcinoma', 'adenocarcinoma', 'sarcoma', 'lymphoma', 'metastasis',
            'neoplasm', 'tumor', 'lesion', 'mass', 'cyst', 'polyp', 'nodule', 'scars', 'adult',
            'dome-shaped', 'flap', 'prophylactic',
            
            # Descriptive terms
            'focal', 'diffuse', 'acute', 'chronic', 'mild', 'moderate', 'severe',
            'benign', 'malignant', 'invasive', 'metastatic', 'astrocytomas', 'anaplastic', 'cataractous',
            
            "mm.", "cm.", "%", "mm,", "cm,", "%,",
            
            # Add measurement combinations
            *[f"{i}mm" for i in range(1, 101)],  # 1mm to 100mm
            *[f"{i}cm" for i in range(1, 51)],   # 1cm to 50cm
            *[f"{i}%" for i in range(1, 100)],   # 1cm to 50cm
            *[f"{i}.{j}" for i in range(10) for j in range(10)],  # Common decimal numbers
            *[f"{i}.{j}cm" for i in range(10) for j in range(10)],
            *[f"{i}.{j}mm" for i in range(10) for j in range(10)],
            *[f"{i}.{j}%" for i in range(10) for j in range(10)],
            *[f"{i}th" for i in range(1, 101)],
        ])
        
        # Map dataset names to cleaning functions
        self.clean_report_map = {
            'iu_xray': self.clean_report_iu_xray,
            'wsi_report': self.clean_report_pathology,
            'mimic_cxr': self.clean_report_mimic_cxr
        }
        self.clean_report = self.clean_report_map.get(self.dataset_name, self.clean_report_mimic_cxr)
        
        self.preserve_case = (self.dataset_name == 'wsi_report')
        # self.ann = json.loads(open(self.ann_path, 'r').read())
        

    # Load or create vocabulary
        if self.vocab_path and os.path.exists(self.vocab_path):
            print(f"INFO: Loading vocabulary from: {self.vocab_path}")
            self.token2idx, self.idx2token = self.load_vocabulary(self.vocab_path)
            print(f"INFO: Vocabulary loaded. Size: {len(self.token2idx)}")
            self.vocabulary_was_built = False
        else:
            if not hasattr(args, 'ann_path') or not args.ann_path:
                 raise ValueError("Cannot build vocabulary: args.ann_path is not provided and vocab_path is missing or invalid.")
            self.ann_path = args.ann_path # Store ann_path only if building vocab
            print(f"INFO: Building vocabulary from annotation file: {self.ann_path}")
            if self.vocab_path: # vocab_path was given but file not found
                print(f"WARNING: vocab_path '{self.vocab_path}' was provided but file not found. Building new vocab from {self.ann_path}.")
            
            try:
                with open(self.ann_path, 'r', encoding='utf-8') as f:
                    self.ann = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Annotation file not found at {self.ann_path} while trying to build vocabulary.")
            except json.JSONDecodeError:
                raise json.JSONDecodeError(f"Error decoding JSON from {self.ann_path} while trying to build vocabulary.")

            self.token2idx, self.idx2token = self.create_vocabulary()
            self.vocabulary_was_built = True
            print(f"INFO: Vocabulary built. Size: {len(self.token2idx)}")


    def load_vocabulary(self, vocab_path_to_load): # Renamed to avoid conflict with self.vocab_path
        """Loads the vocabulary from a JSON file."""
        try:
            with open(vocab_path_to_load, 'r', encoding='utf-8') as f:
                data = json.load(f)
            token2idx = data['token2idx']
            idx2token = {int(k): v for k, v in data['idx2token'].items()} # Ensure keys are int
            return token2idx, idx2token
        except Exception as e:
            print(f"ERROR: Could not load vocabulary from {vocab_path_to_load}: {e}")
            raise

    def save_vocabulary(self, vocab_path_to_save): # Renamed to avoid conflict
        """Saves the current vocabulary to a JSON file."""
        if not self.token2idx or not self.idx2token:
            print("ERROR: Vocabulary not built or loaded properly yet. Cannot save.")
            return
        try:
            if os.path.dirname(vocab_path_to_save): # Ensure directory exists only if path includes a directory
                 os.makedirs(os.path.dirname(vocab_path_to_save), exist_ok=True)
            with open(vocab_path_to_save, 'w', encoding='utf-8') as f:
                json.dump({'token2idx': self.token2idx,
                           'idx2token': {str(k): v for k, v in self.idx2token.items()}}, # Keys in JSON must be strings
                          f, indent=4)
            print(f"INFO: Vocabulary saved to {vocab_path_to_save}")
        except Exception as e:
            print(f"ERROR: Could not save vocabulary to {vocab_path_to_save}: {e}")
            raise

    def create_vocabulary(self):
        """Creates vocabulary from the 'train' split of the annotation file (self.ann)."""
        total_tokens = []
        if 'train' not in self.ann: # self.ann should be loaded before calling this
            raise KeyError(f"Annotation data (self.ann) must contain a 'train' key for vocabulary creation.")

        for example in self.ann['train']:
            if 'report' not in example:
                continue
            tokens = self.clean_report(example['report']).split()
            total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        vocab_tokens = set(k for k, v in counter.items() if v >= self.threshold)
        vocab_tokens.update(self.domain_vocab)
        
        # Ensure special tokens are handled correctly and are at the beginning
        # and that vocab_tokens doesn't include them before sorting
        processed_vocab_tokens = sorted(list(v for v in vocab_tokens if v not in [self.pad_token, self.bos_token, self.eos_token, self.unk_token]))
        
        vocab = [self.pad_token, self.bos_token, self.eos_token, self.unk_token] + processed_vocab_tokens

        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for idx, token in enumerate(vocab)}
        return token2idx, idx2token

    def normalize_whitespace(self, text):
        if self.dataset_name == 'wsi_report':
            text = re.sub(r'\n\s*\n', '\n', text)
            text = re.sub(r' +', ' ', text)
        else:
            text = ' '.join(text.split())
            text = text.replace('\n', ' ')
        return text

    def normalize_punctuation(self, text):
        # Replace multiple periods with single period, but preserve decimal points
        text = re.sub(r'\.(?![\d])', '. ', text)  # Add space after period unless followed by digit
        text = re.sub(r'\.{2,}', '.', text)       # Replace multiple periods
        
        if self.dataset_name == 'wsi_report':
            # Preserve commas in numbers (e.g., 1,000)
            text = re.sub(r'([^0-9]),([^0-9])', r'\1, \2', text)  # Add spaces around non-numeric commas
            text = re.sub(r'([.,;:])\1+', r'\1', text)
        else:
            text = re.sub(r'\d+\.\s*', '', text)
            text = re.sub(r'\s*\.\s*', '. ', text)
        
        return text.strip()

    def clean_sentence(self, sentence):
        if self.dataset_name == 'wsi_report':
            # Preserve numbers, decimal points, and measurements
            # Allow numbers, letters, basic punctuation, and measurement units
            sentence = re.sub(r'[^a-zA-Z0-9\s.,;:()\-\.]', ' ', sentence)
            
            # Ensure measurements stay together (no space between number and unit)
            sentence = re.sub(r'(\d+)\s+(mm|cm|ml|cc|µm)', r'\1\2', sentence)
            
            # Normalize spaces
            sentence = re.sub(r'\s+', ' ', sentence)
            cleaned = sentence.strip()
        else:
            sentence = re.sub(r'[^a-zA-Z0-9\s.,;:]', ' ', sentence)
            cleaned = sentence.lower().strip()
            cleaned = re.sub(r'[.,;:]', '', cleaned)
        
        return cleaned

    def tokenize_with_numbers(self, text):
        """Special tokenization that preserves numbers and measurements."""
        # Split on spaces but preserve special patterns
        tokens = []
        words = text.split()
        
        for word in words:
            # Check if word contains a number pattern
            if re.search(r'\d', word):
                # Keep measurements together (e.g., "7mm", "4.5cm")
                if re.match(r'^\d+\.?\d*(?:mm|cm|ml|cc|µm)?$', word):
                    tokens.append(word)
                else:
                    # Split other numeric patterns
                    numeric_parts = re.findall(r'(\d+\.?\d*)|([^0-9]+)', word)
                    tokens.extend(part for parts in numeric_parts for part in parts if part)
            else:
                tokens.append(word)
        
        return tokens

    def __call__(self, report):
        """Convert a report to token ids."""
        cleaned_report = self.clean_report(report)
        
        if self.dataset_name == 'wsi_report':
            tokens = self.tokenize_with_numbers(cleaned_report)
        else:
            tokens = cleaned_report.split()
        
        tokens = [self.bos_token] + tokens + [self.eos_token]
        return [self.get_id_by_token(token) for token in tokens]

    def clean_report_pathology(self, report):
        """Special handling for pathology reports."""
        # Normalize underscores and whitespace
        report = re.sub(r'_+', ' ', report)
        report = self.normalize_whitespace(report)
        report = self.normalize_punctuation(report)
        
        # Split on periods while preserving them
        sentences = re.split(r'(?<=\.)\s+', report)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Clean each sentence
        cleaned_sentences = [self.clean_sentence(sent) for sent in sentences]
        cleaned_sentences = [sent for sent in cleaned_sentences if sent]
        
        # Join with proper punctuation
        if cleaned_sentences:
            # Add period if not present
            result = '. '.join(s.rstrip('.') for s in cleaned_sentences) + '.'
            # Clean up any double periods
            result = re.sub(r'\.+', '.', result)
            return result
        return ''

    def clean_report_iu_xray(self, report):
        # Normalize whitespace and punctuation
        report = self.normalize_whitespace(report)
        report = self.normalize_punctuation(report)
        
        # Split into sentences and clean each one
        sentences = [s.strip() for s in report.split('.') if s.strip()]
        cleaned_sentences = [self.clean_sentence(sent) for sent in sentences]
        cleaned_sentences = [sent for sent in cleaned_sentences if sent]
        
        # Join sentences with proper spacing
        return ' . '.join(cleaned_sentences) + ' .'

    def clean_report_mimic_cxr(self, report):
        # Additional preprocessing for MIMIC-CXR
        report = re.sub(r'_+', ' ', report)  # Replace underscores with space
        report = self.normalize_whitespace(report)
        report = self.normalize_punctuation(report)
        
        sentences = [s.strip() for s in report.split('.') if s.strip()]
        cleaned_sentences = [self.clean_sentence(sent) for sent in sentences]
        cleaned_sentences = [sent for sent in cleaned_sentences if sent]
        
        return ' . '.join(cleaned_sentences) + ' .'

    def get_token_by_id(self, id):
        return self.idx2token.get(id, self.unk_token)

    def get_id_by_token(self, token):
        # For non-WSI reports, convert to lowercase for token lookup
        if not self.preserve_case:
            token = token.lower()
        return self.token2idx.get(token, self.token2idx[self.unk_token])

    def get_vocab_size(self):
        return len(self.token2idx)

    def decode(self, ids):
        """Convert token ids back to text."""
        tokens = []
        for idx in ids:
            if idx == self.token2idx[self.eos_token]:
                break
            if idx != self.token2idx[self.pad_token]:
                token = self.idx2token[idx]
                if token != self.bos_token:  # Skip BOS token in output
                    tokens.append(token)
        
        # Join tokens with proper spacing
        if self.dataset_name == 'wsi_report':
            # For WSI reports, handle punctuation specially
            text = ''
            for i, token in enumerate(tokens):
                if token in '.,;:':
                    text = text.rstrip() + token + ' '
                else:
                    text += token + ' '
            return text.strip()
        else:
            return ' '.join(tokens)

    def decode_batch(self, ids_batch):
        """Convert a batch of token ids back to texts."""
        return [self.decode(ids) for ids in ids_batch]


import torch
from transformers import AutoTokenizer
import os

class HuggingFaceTokenizer(object):
    """
    A simplified and robust tokenizer that loads the Llama3-OpenBioLLM-8B
    tokenizer from the Hugging Face Hub every time. It does not save to disk.
    """
    def __init__(self, args):
        """
        Initializes the tokenizer by loading it directly from the Hub.
        'args' should have 'max_seq_length'.
        The 'args.vocab_path' is no longer used by this class.
        """
        model_name = "aaditya/Llama3-OpenBioLLM-8B"
        print(f"INFO: Loading tokenizer directly from Hugging Face Hub: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = args.max_seq_length

        # Set the PAD token if it's not already set.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"INFO: Tokenizer 'pad_token' not set. Using 'eos_token' as 'pad_token'.")

        # --- FIX for AttributeError ---
        # Create .idx2token and .token2idx attributes for backward compatibility with your model.
        print("INFO: Creating .idx2token and .token2idx attributes for model compatibility.")
        self.token2idx = self.tokenizer.get_vocab()
        self.idx2token = {id: token for token, id in self.token2idx.items()}

    def __call__(self, report: str) -> list[int]:
        """Encodes a single report string into a list of token IDs."""
        encoded = self.tokenizer.encode(
            report,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True
        )
        return encoded

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs back into a clean string."""
        decoded_text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return decoded_text.strip()

    def decode_batch(self, ids_batch: list[list[int]]) -> list[str]:
        """Decodes a batch of token ID lists efficiently."""
        return self.tokenizer.batch_decode(ids_batch, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """Returns the total size of the vocabulary."""
        return len(self.tokenizer)

    # The save_vocabulary method is now removed.
    # def save_vocabulary(self, vocab_path: str):
    #     pass