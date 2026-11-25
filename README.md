# MRIS: Multimodal Reality Implicit Safety

This repository provides the source code and benchmark (MRISbench) for evaluating and mitigating Multimodal Reality Implicit Safety (MRIS) risks in vision-language models. MRIS captures scenarios where seemingly safe text queries become dangerous when combined with specific visual contexts.

We demonstrate our approach using Qwen2.5-VL-7B-Instruct and Llama-3.2-11B-Vision-Instruct as evaluation models.

> **Content Warning:** This repository contains materials related to AI safety evaluation, including examples involving sensitive topics such as self-harm, violence, child safety, illegal activities, and other categories of harmful content. These materials are provided strictly for research purposes to advance the safety of multimodal AI systems. Reader discretion is advised.

---

## Repository Structure

```
MRIS/
├── README.md                        # This file
├── env.yml                          # Conda environment specification
├── process_dataset.py               # Dataset path preprocessing script
├── open_source/                     # MRISbench Evaluation Module
│   ├── multimodal/                  # Multimodal (text + image) evaluation
│   │   ├── run_multimodal_single.sh
│   │   └── utils/
│   └── text/                        # Text-only evaluation
│       ├── run_text_single.sh
│       └── utils/
└── GhostGuider/                     # GhostGuider Steering Module
    ├── Llama/                       # Llama-Vision implementation
    └── qwen/                        # Qwen-VL implementation
```

---

## Environment Setup

### Step 1: Create Conda Environment

```bash
conda env create -f env.yml
conda activate env
```

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Ensure the output shows `CUDA available: True`. If not, check your CUDA driver and PyTorch installation.

---

## Model Weights Download

Download the required models from Hugging Face:

- **Qwen2.5-VL-7B-Instruct**: [Model Page](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- **Llama-3.2-11B-Vision-Instruct**: [Model Page](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)

```bash
# Example using huggingface-cli
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /path-to-model/Qwen2.5-VL-7B-Instruct
huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct --local-dir /path-to-model/Llama-3.2-11B-Vision-Instruct
```

---

## Dataset Download

On our shared anonymous [Google Drive](https://drive.google.com/drive/folders/1o7-_GiT7pSk2iDJ9er7AfW_WJkG58dzs?usp=sharing), download the following compressed file:
- `MRIS_Data.tar.gz`

### Step 1: Extract Data

Place `MRIS_Data.tar.gz` under the repository root, then extract it and rename the extracted folder to `data/`:

```bash
tar -xzf MRIS_Data.tar.gz
mv MRIS_Data data
```

The resulting `data/` folder will have the following structure:

```
data/
├── benign_original.json
├── malicious_original.json
├── benign/
├── malicious/
├── Direct_VLSafe/
└── MRISbench/
    ├── MRISbench_original.json
    ├── MRISbench_test_original.json
    ├── safe_image_compress/
    └── unsafe_image_compress/
```

### Step 2: Preprocess Dataset Paths

Run the preprocessing script to update image paths in the JSON files. Replace `/path-to-code/MRIS/` with the absolute path to this repository:

```bash
python process_dataset.py --prefix=/path-to-code/MRIS/
```

This generates processed JSON files (without the `_original` suffix) with correct absolute paths.

---

## Evaluating Models on MRISbench

This section evaluates text-only and multimodal LLMs on the MRISbench benchmark.

### Step 1: Configure Paths

Edit the paths in the following scripts to point to your model and data locations:
- `open_source/multimodal/run_multimodal_single.sh`
- `open_source/text/run_text_single.sh`

In each script, update:

| Variable | Description |
| --- | --- |
| `--model_path` | Absolute path to the model weights |
| `--input_file` | Absolute path to `data/MRISbench/MRISbench.json` |
| `--output_record_dir` | Output directory for per-sample records |
| `--output_logs_dir` | Output directory for logs |
| `--output_record_stats` | Output directory for aggregated statistics |

### Step 2: Run Evaluation

```bash
cd open_source

# Multimodal evaluation
chmod +x ./multimodal/run_multimodal_single.sh && ./multimodal/run_multimodal_single.sh

# Text-only evaluation
chmod +x ./text/run_text_single.sh && ./text/run_text_single.sh
```

### Output Files

After completion, results are organized as follows (example for multimodal; `data/text` has the same structure):

```
data/multimodal/
├── logs/
│   ├── Llama-3.2-11B-Vision-Instruct.log
│   └── Qwen2.5-VL-7B-Instruct.log
├── records/
│   ├── Llama-3.2-11B-Vision-Instruct/
│   │   ├── *_results_Aware_Answer.json
│   │   ├── *_results_Aware_Refuse.json
│   │   ├── *_results_Ignore_Answer.json
│   │   ├── *_results_Ignore_Refuse.json
│   │   └── results.json
│   └── Qwen2.5-VL-7B-Instruct/
│       └── ... (same structure)
└── stats/
    ├── Llama-3.2-11B-Vision-Instruct.json
    └── Qwen2.5-VL-7B-Instruct.json
```

| File | Description |
| --- | --- |
| `records/<model>/*_results_*.json` | Per-sample model responses for each choice type |
| `records/<model>/results.json` | Aggregated evaluation results |
| `stats/<model>.json` | Summary statistics |
| `logs/<model>.log` | Execution logs |

---

## Running GhostGuider

GhostGuider is an activation steering method that enhances MRIS risk recognition while provably preserving alignment on benign and explicitly harmful inputs. Below are two examples demonstrating how to apply GhostGuider to Llama-Vision and Qwen-VL on MRISbench. 

### Example 1: Llama-Vision

```bash
cd GhostGuider/Llama
```

#### Step 1: Compute Risk Vectors

First, update the `image_mode_paths` and `text_mode_paths` variables in `llama_risk_vector.py` to point to the evaluation outputs from the previous section. Then run:

```bash
python llama_risk_vector.py --model_path=/path-to-model/Llama-3.2-11B-Vision-Instruct
```

Output in `risk_vector/`:

| File | Description |
| --- | --- |
| `risk_vectors_llama32_instruct_mean_diff.npy` | Mean difference risk vector (used in Step 2) |
| `risk_vectors_llama32_instruct_class_a.npy` | Safe class activation centroids |
| `risk_vectors_llama32_instruct_class_b.npy` | Unsafe class activation centroids |
| `risk_vectors_llama32_instruct.json` | Metadata |

#### Step 2: Compute Steering Matrices

Edit `llama_run_compute_vlm_steering.sh` and update the following parameters:

| Variable | Description | Example |
| --- | --- | --- |
| `MODEL_PATH` | Absolute path to Llama-Vision model | `/path-to-model/Llama-3.2-11B-Vision-Instruct` |
| `BENIGN_DATA` | Path to benign data | `/path-to-code/MRIS/data/benign.json` |
| `HARMFUL_DATA` | Path to malicious data | `/path-to-code/MRIS/data/malicious.json` |
| `RISK_VECTORS` | Path to Step 1's risk vector output | `./risk_vector/risk_vectors_llama32_instruct_mean_diff.npy` |

Then run:

```bash
chmod +x llama_run_compute_vlm_steering.sh && ./llama_run_compute_vlm_steering.sh
```

Output in `outputs/llama_dual_protected_steering/`:

| File | Description |
| --- | --- |
| `steering_matrices_instruct.pt` | Steering matrices for inference |
| `steering_matrices_instruct_full.pt` | Full matrices with diagnostic info |
| `steering_matrices_instruct_metadata.json` | Computation metadata |

#### Step 3: Evaluate with Steering

Edit `run_evaluation_2.sh` and update:

| Variable | Description | Example |
| --- | --- | --- |
| `MODEL_PATH` | Absolute path to Llama-Vision model | `/path-to-model/Llama-3.2-11B-Vision-Instruct` |
| `STEERING_CHECKPOINT` | Path to Step 2's steering matrices | `./outputs/llama_dual_protected_steering/steering_matrices_instruct.pt` |
| `INPUT_FILE` | Path to MRISbench test set | `/path-to-code/MRIS/data/MRISbench/MRISbench_test.json` |

Then run:

```bash
chmod +x run_evaluation_2.sh && ./run_evaluation_2.sh
```

Results are saved in `results/`:

```
results/
├── logs/
│   └── LlamaVision_steering_0.6.log
├── records/
│   └── LlamaVision_steering_0.6/
│       ├── *_results_Aware_Answer.json
│       ├── *_results_Aware_Refuse.json
│       ├── *_results_Ignore_Answer.json
│       ├── *_results_Ignore_Refuse.json
│       └── results.json
└── stats/
    └── LlamaVision_steering_0.6.json
```

---

### Example 2: Qwen2.5-VL

```bash
cd GhostGuider/qwen
```

#### Step 1: Compute Risk Vectors

Update `image_mode_paths` and `text_mode_paths` in `qwen_risk_vector.py`, then run:

```bash
python qwen_risk_vector.py --model_path=/path-to-model/Qwen2.5-VL-7B-Instruct
```

Output in `risk_vector/`:

| File | Description |
| --- | --- |
| `risk_vectors_qwen2.5_instruct_mean_diff.npy` | Mean difference risk vector (used in Step 2) |
| `risk_vectors_qwen2.5_instruct_class_a.npy` | Safe class activation centroids |
| `risk_vectors_qwen2.5_instruct_class_b.npy` | Unsafe class activation centroids |
| `risk_vectors_qwen2.5_instruct.json` | Metadata |

#### Step 2: Compute Steering Matrices

Edit `qwen_run_compute_vlm_steering.sh` and update:

| Variable | Description | Example |
| --- | --- | --- |
| `MODEL_PATH` | Absolute path to Qwen-VL model | `/path-to-model/Qwen2.5-VL-7B-Instruct` |
| `BENIGN_DATA` | Path to benign data | `/path-to-code/MRIS/data/benign.json` |
| `HARMFUL_DATA` | Path to malicious data | `/path-to-code/MRIS/data/malicious.json` |
| `RISK_VECTORS` | Path to Step 1's risk vector output | `./risk_vector/risk_vectors_qwen2.5_instruct_mean_diff.npy` |

Then run:

```bash
chmod +x qwen_run_compute_vlm_steering.sh && ./qwen_run_compute_vlm_steering.sh
```

Output in `outputs/qwen_dual_protected_steering/`:

| File | Description |
| --- | --- |
| `steering_matrices_instruct.pt` | Steering matrices for inference |
| `steering_matrices_instruct_full.pt` | Full matrices with diagnostic info |
| `steering_matrices_instruct_metadata.json` | Computation metadata |

#### Step 3: Evaluate with Steering

Edit `run_evaluation_2.sh` and update:

| Variable | Description | Example |
| --- | --- | --- |
| `MODEL_PATH` | Absolute path to Qwen-VL model | `/path-to-model/Qwen2.5-VL-7B-Instruct` |
| `STEERING_CHECKPOINT` | Path to Step 2's steering matrices | `./outputs/qwen_dual_protected_steering/steering_matrices_instruct.pt` |
| `INPUT_FILE` | Path to MRISbench test set | `/path-to-code/MRIS/data/MRISbench/MRISbench_test.json` |

Then run:

```bash
chmod +x run_evaluation_2.sh && ./run_evaluation_2.sh
```

Results are saved in `results/`:

```
results/
├── logs/
│   └── qwen2.5_steering_0.7_all.log
├── records/
│   └── qwen2.5_steering_0.7_all/
│       ├── *_results_Aware_Answer.json
│       ├── *_results_Aware_Refuse.json
│       ├── *_results_Ignore_Answer.json
│       ├── *_results_Ignore_Refuse.json
│       └── results.json
└── stats/
    └── qwen2.5_steering_0.7_all.json
```

---

## Ethical Considerations

This research is intended to improve the safety and robustness of multimodal AI systems by identifying and mitigating implicit safety risks that emerge from the combination of text and visual inputs.

**Content Warning:** The MRISbench dataset and the paper contain examples involving sensitive topics, including but not limited to: self-harm, violence and extremism, child safety, illegal activities, privacy violations, misinformation, harassment and discrimination, dangerous behavior, and inappropriate content. These examples are included solely for the purpose of evaluating and improving AI safety mechanisms. They do not represent the views or endorsements of the authors.

**Responsible Use:** This repository and its associated dataset are intended exclusively for AI safety research. Any use for malicious purposes, including exploiting identified vulnerabilities to cause harm, is strictly prohibited. Researchers working with these materials are advised to take appropriate precautions regarding exposure to sensitive content, including limiting session durations and utilizing available support resources.
