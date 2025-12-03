# SAEMark

Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders

## Overview

SAEMark generates watermarks while maintaining text quality. It samples multiple candidates and selects those meeting the specified criteria.

## Prerequisites

### TGI Server Setup

Before running the generator, you need to start the TGI (Text Generation Interface) server:

```bash
python inference.py \
  -d /path/to/your/base/model \
  -g <gpu-number> \
  -p <port>
```

This command initializes the TGI server with your base model. The server must be running before executing the generator script.

Parameters:
- `-d`: Path to your base language model
- `-g`: GPU number to use for the server
- `-p`: Port number for the TGI server (use this same port in generator.py)

You could also use `vllm` and `vllm serve` instead of using `text-generation-inference` from huggingface.

## Datasets

The following datasets are available in the `dataset` directory:

> Note: In dataset filenames, the numerical suffix indicates the number of different users who generated watermarks:
> - `_1`: Individual user generations
> - `_1024`: Dataset for 1024 different users, each with their own watermark key

### Dataset Format

Each dataset file follows a JSONL (JSON Lines) format, where each line contains a JSON object with the following fields:
- `id`: Unique identifier for the data point (e.g., "c4/1")
- `prompt`: The text prompt used for generation
- `target_key`: The key used for generating watermarked text

Example entry:
```
{
  "id": "c4/1",
  "prompt": "Once upon a time...",
  "target_key": "1"
}
```

- **c4**: English text generation dataset
  - Contains high-quality English web text for generating semantic variations
  - File format: JSONL
  - Examples: 
    - `dataset/c4/c4_1.jsonl` (for single user)
    - `dataset/c4/c4_1024.jsonl` (for 1024 different users)

- **lcsts**: Chinese text generation dataset
  - Used for generating semantic variations in Chinese
  - File format: JSONL
  - Examples:
    - `dataset/lcsts/lcsts_1.jsonl` (for single user)
    - `dataset/lcsts/lcsts_1024.jsonl` (for 1024 different users)

- **mbpp**: Code generation dataset
  - Specialized for programming code variations
  - Contains various programming language samples
  - File format: JSONL
  - Example: `dataset/mbpp/mbpp_1.jsonl` (for watermark key #1)

- **pandalm**: Text quality comparison dataset
  - Used as a reference for evaluating generation quality
  - Helps in assessing the text quality of the watermark
  - File format: JSONL
  - Example: `dataset/pandalm/pandalm_1.jsonl` (for watermark key #1)

## Generator

Basic command structure:

```bash
python watermarked_generator.py [options]  # For watermarked generation
python unwatermarked_generator.py [options]  # For unwatermarked generation
```

### Unwatermarked Generation

For generating baseline unwatermarked text, use `unwatermarked_generator.py`. Examples for different languages:

#### Output Format

The generated output is saved in JSONL format with the following fields:
- `id`: The same identifier as in the input dataset
- `prompt`: The original generation prompt
- `target_key`: The key used for generation 
- `unwatermarked`: The generated text without watermark application

Example output:
```
{
  "id": "c4/1",
  "prompt": "Once upon a time...",
  "target_key": "1",
  "unwatermarked": "Once upon a time, in a land far, far away..."
}
```

```bash
# English text generation
python unwatermarked_generator.py \
  --dataset 'dataset/c4/c4_1.jsonl' \
  --start 0 \
  --end 125 \
  --base /path/to/your/base/model \
  --candidates 50 \
  --units 10 \
  --ip <tgi-server-ip> \
  --port <tgi-server-port> \
  --output_path <output-path> \
  --lang 'en'

# Chinese text generation
python unwatermarked_generator.py \
  --dataset 'dataset/lcsts/lcsts_1.jsonl' \
  --start 0 \
  --end 125 \
  --base /path/to/your/base/model \
  --candidates 50 \
  --units 10 \
  --ip <tgi-server-ip> \
  --port <tgi-server-port> \
  --output_path <output-path> \
  --lang 'zh'

# Code generation
python unwatermarked_generator.py \
  --dataset 'dataset/mbpp/mbpp_1.jsonl' \
  --start 0 \
  --end 125 \
  --base /path/to/your/base/model \
  --candidates 50 \
  --units 20 \
  --ip <tgi-server-ip> \
  --port <tgi-server-port> \
  --output_path <output-path> \
  --lang 'code'
```

Note: Unwatermarked generation uses a simplified parameter set compared to watermarked generation, as it doesn't require watermark-specific parameters like `--mask`, `--mu`, `--sigma`, etc.

### Required Parameters

- `--dataset`: Path to input dataset (JSONL format)
- `--base`: Path to base language model
- `--anchor`: Path to anchor language model
- `--device`: GPU device to use (e.g., 'cuda:0')
- `--output_path`: Output file path for results

### Optional Parameters

- `--start`: Starting index in dataset (default: 0)
- `--end`: Ending index in dataset
- `--mask`: Whether to add background frequent feature mask (default: True)
- `--mu`: Target mean similarity value (default: 0.13)
- `--sigma`: Standard deviation for similarity (default: 0.02)
- `--rmin`: Minimum similarity range (default: 0.95)
- `--rmax`: Maximum similarity range (default: 1.05)
- `--omin`: Minimum overlap ratio (default: 0.95)
- `--lang`: Language mode - 'en' (English), 'zh' (Chinese), 'code' (Programming)
- `--candidates`: Number of candidates to sample (default: 50)
- `--units`: Number of generation iterations (default: 10)
- `--attempts`: Maximum retry attempts (default: 5)
- `--ip`: TGI server IP address
- `--port`: TGI server port

## Example

Note:
- The `--mask` parameter enables background frequent feature masking to improve watermark quality
- For ablation study, you can:
  - Set `--mask False` to disable feature masking
  - Modify `--mu`, `--sigma`, `--rmin`, `--rmax`, `--omin` to recover the results in our paper

### English Generation

The following example shows how to generate English text variations:

```bash
python watermarked_generator.py \
  --dataset 'dataset/c4/c4_1.jsonl' \
  --start 0 \
  --end 125 \
  --mask True \
  --base /path/to/your/base/model \
  --anchor /path/to/your/anchor/model \
  --device <cuda-device> \
  --mu 0.13 \
  --sigma 0.02 \
  --rmin 0.95 \
  --rmax 1.05 \
  --omin 0.95 \
  --lang 'en' \
  --candidates 50 \
  --units 10 \
  --attempts 5 \
  --ip <tgi-server-ip> \
  --port <tgi-server-port> \
  --output_path <output-path>
```

#### Output Format

The generated output is saved in JSONL format with the following fields:
- `id`: The same identifier as in the input dataset
- `prompt`: The original generation prompt
- `target_key`: The key used for generation 
- `watermarked`: The generated text with watermark application

Example output:
```
{
  "id": "c4/1",
  "prompt": "Once upon a time...",
  "target_key": "1",
  "watermarked": "Once upon a time, in a land far, far away..."
}
```

Note: You can also generate PandaLM-style evaluation data using the same command structure by simply changing the input dataset:

```bash
python watermarked_generator.py \
  --dataset 'dataset/pandalm/pandalm_1.jsonl' \
  --start 0 \
  --end 125 \
  --mask True \
  --base /path/to/your/base/model \
  --anchor /path/to/your/anchor/model \
  --device <cuda-device> \
  --mu 0.13 \
  --sigma 0.02 \
  --rmin 0.95 \
  --rmax 1.05 \
  --omin 0.95 \
  --lang 'en' \
  --candidates 50 \
  --units 10 \
  --attempts 5 \
  --ip <tgi-server-ip> \
  --port <tgi-server-port> \
  --output_path <output-path>
```

### Chinese Generation

For Chinese text watermarking, use the LCSTS dataset and set the language mode to 'zh'. The process is similar to English generation.

```bash
python watermarked_generator.py \
  --dataset 'dataset/lcsts/lcsts_1.jsonl' \
  --start 0 \
  --end 125 \
  --mask True \
  --base /path/to/your/base/model \
  --anchor /path/to/your/anchor/model \
  --device <cuda-device> \
  --mu 0.13 \
  --sigma 0.02 \
  --rmin 0.95 \
  --rmax 1.05 \
  --omin 0.95 \
  --lang 'zh' \
  --candidates 50 \
  --units 10 \
  --attempts 5 \
  --ip <tgi-server-ip> \
  --port <tgi-server-port> \
  --output_path <output-path>
```

Note: For Chinese generation, make sure to use language models that support Chinese text processing. The LCSTS dataset contains high-quality Chinese text samples suitable for watermark generation.

### Code Generation

For programming code generation, use the MBPP dataset with adjusted parameters for code-specific requirements:

```bash
python watermarked_generator.py \
  --dataset 'dataset/mbpp/mbpp_1.jsonl' \
  --start 0 \
  --end 125 \
  --mask True \
  --base /path/to/your/code/model \
  --anchor /path/to/your/anchor/model \
  --device <cuda-device> \
  --mu 0.13 \
  --sigma 0.02 \
  --rmin 0.8 \    # Note: Different from text generation
  --rmax 1.2 \    # Wider range for code variations
  --omin 0.85 \   # Lower overlap threshold for code
  --lang 'code' \
  --candidates 50 \
  --units 20 \    # More units for code generation
  --attempts 5 \
  --ip <tgi-server-ip> \
  --port <tgi-server-port> \
  --output_path <output-path>
```

Note: Code generation uses different parameters compared to text generation:
- Wider similarity range (`rmin`=0.8, `rmax`=1.2)
- Lower overlap threshold (`omin`=0.85)
- More generation units (`units`=20)
- Requires a code-capable language model

> Note: The output format remains consistent across all language modes (English, Chinese, and code generation). The generated output will always contain the same fields (`id`, `prompt`, `target_key`, and either `watermarked` or `unwatermarked` depending on the generation type).

## Validator

The validator is used to detect watermarks in generated text, particularly for multi-user scenarios. It supports both watermarked and unwatermarked text validation.

> Important: For best results, use the same parameters (`--rmin`, `--rmax`, `--omin`) in validation as were used during generation. This ensures consistent detection of watermarks based on the original generation criteria.

### Examples

For multi-user validation:
```bash
python validator.py \
  --input_path 'tmp_en_8192.jsonl' \
  --watermarked True \
  --start 0 \
  --end 10 \
  --mask True \
  --anchor /path/to/anchor/model \
  --device cuda:0 \
  --mu 0.13 \
  --sigma 0.02 \
  --rmin 0.95 \
  --rmax 1.05 \
  --omin 0.95 \
  --lang en \
  --task multi \
  --users 8192 \
  --output_path 8192.jsonl
```

### Output Format

The multi-user validation output is saved in JSONL format with the following fields:
- `id`: The same identifier as in the input dataset
- `prompt`: The original generation prompt
- `target_key`: The key used for generation
- `watermarked`/`unwatermarked`
- `detected_key`: The detected watermark key

A successful watermark detection occurs when `detection_key` matches `target_key`.

Example output:
```
{
  "id": "c4/1",
  "prompt": "Once upon a time...",
  "target_key": "1",
  "watermarked": "Once upon a time, in a land far, far away...",
  "detected_key": "1"
}
```

For ROC curve analysis:
```bash
python validator.py \
  --input_path 'tmp_en.jsonl' \
  --watermarked True \
  --start 0 \
  --end 2 \
  --mask True \
  --anchor /path/to/anchor/model \
  --device cuda:0 \
  --mu 0.13 \
  --sigma 0.02 \
  --rmin 0.95 \
  --rmax 1.05 \
  --omin 0.95 \
  --lang en \
  --task roc \
  --output_path roc_en.jsonl
```

### ROC Analysis Output Format

The ROC curve analysis results are saved in JSONL format with the following fields:
- `label`: Binary indicator where:
  - `1`: Text contains watermark
  - `0`: Text is unwatermarked
- `score`: P-value from t-test detection (lower scores indicate stronger watermark presence)

Example output:
```
{
  "label": 1,
  "score": 0.00022646967299703592  # Watermarked text with strong detection
}
{
  "label": 0,
  "score": 1.0  # Unwatermarked text
}
```

Important Parameters:
- `--watermarked`: Set to True for detecting watermarked text, False for unwatermarked text
- `--users`: Number of users in the watermarking system (only required for `--task multi`, should match the dataset suffix, e.g., use 1024 for 'c4_1024.jsonl')
- `--task`: Use 'multi' for multi-user validation or 'roc' for ROC curve analysis

## Result Analysis

After validation, use `get_result.py` to analyze the results of watermark detection. This script supports multiple analysis tasks including multi-user test analysis and ROC curve generation.

> Important: For ROC curve analysis, we strongly recommend combining both watermarked and unwatermarked validation results in a single file before analysis. This ensures accurate comparison and better visualization of the detection performance.

### Multi-User Test Analysis

To analyze the results of a multi-user watermark detection test:

```bash
python get_result.py \
  --input_path '8192.jsonl' \
  --start 0 \
  --end 10 \
  --task multi
```

Parameters:
- `--input_path`: Path to the validation results file
- `--start`: Starting index in results file
- `--end`: Ending index in results file
- `--task`: Analysis task type ('multi' for multi-user analysis, 'roc' for ROC curve generation)
- `--output_path`: Required only for ROC curve generation, specifies where to save the plot

The script will output the accuracy percentage of correct user identification from watermarked text. Example output:
```
The result for this multi test is 95.0
```

### ROC Curve Analysis

To generate an ROC curve from the validation results:

```bash
python get_result.py \
  --input_path roc_c4.jsonl \
  --start 0 \
  --end 1000 \
  --task roc \
  --output_path roc.png
```

The script will:
1. Calculate and display detection metrics (Precision, Recall, F1-score, Accuracy)
2. Generate the ROC curve plot and save it as a PNG file
3. Print the TPR (True Positive Rate) at FPR ≈ 0.01

Note: Make sure your input file contains both watermarked and unwatermarked validation results for proper ROC curve generation.

## Attack
The attack output maintains the same format as the generation output, containing fields:
- `id`: The same identifier as in the input dataset
- `prompt`: The original generation prompt
- `target_key`: The key used for generation
- `watermarked`/`unwatermarked`: **The attacked version of the text**, maintaining the same label as the input

Example output:
```
{
  "id": "c4/1",
  "prompt": "Once upon a time...",
  "target_key": "1",
  "watermarked": "Once upon a time, in a land far, far away..."
}
```

### Basic Usage

```bash
python attack.py \
  --input_path <input-file> \
  --watermarked <True/False> \
  --ratio <modification-ratio> \
  --output_path <output-file> \
  --attack_type <attack-type>
```

Parameters:
- `--input_path`: Path to input data file
- `--watermarked`: Whether to attack watermarked text (True) or unwatermarked text (False)
- `--ratio`: Ratio of words to modify (float between 0 and 1)
- `--output_path`: Path for saving attacked text
- `--attack_type`: Attack method to use:
  - `deletion`: Random word deletion
  - `syno_sub`: Simple synonym substitution
  - `context_sub`: Context-aware synonym substitution (requires BERT model)

### Examples

Word deletion attack:
```bash
python attack.py \
  --input_path input.jsonl \
  --watermarked True \
  --ratio 0.1 \
  --output_path output.jsonl \
  --attack_type deletion
```

Simple synonym substitution:
```bash
python attack.py \
  --input_path input.jsonl \
  --watermarked True \
  --ratio 0.1 \
  --output_path output.jsonl \
  --attack_type syno_sub
```

Context-aware synonym substitution:
```bash
python attack.py \
  --input_path input.jsonl \
  --watermarked True \
  --ratio 0.1 \
  --output_path output.jsonl \
  --attack_type context_sub \
  --bert_path /path/to/bert/model \
  --device cuda:0
```

Note: The `--bert_path` and `--device` parameters are only required for context-aware substitution (`context_sub`). They can be omitted for deletion and simple synonym substitution attacks.

