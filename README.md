# LLMCBench: Benchmarking Large Language Model Compression for Efficient Deployment

![image-20241026195404186](./figs/f1.png)

 The **L**arge **L**anguage **M**odel **C**ompression **Bench**mark (LLMCBench) is a rigorously designed benchmark with an in-depth analysis for LLM compression algorithms. 

## Installation

```
git clone https://github.com/AboveParadise/LLMCBench.git
cd LLMCBench

conda create -n llmcbench python=3.9
conda activate llmcbench
pip install -r requirements.txt
```

## Usage

This repo contains codes for testing MMLU, MNLI, QNLI, Wikitext2, advGLUE, TruthfulQA datasets and FLOPs.

#### Testing MMLU

```
bash scripts/run_mmlu.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--data_dir` : Dataset location.
- `--ntrain` : number of shots.
- `--seqlen` : Denotes the maximum input sequence length for LLM.

#### Testing MNLI

```
bash scripts/run_mnli.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--data_dir` : Dataset location.
- `--ntrain` : number of shots.
- `--seqlen` : Denotes the maximum input sequence length for LLM.

#### Testing QNLI

```
bash scripts/run_qnli.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--data_dir` : Dataset location.
- `--ntrain` : number of shots.
- `--seqlen` : Denotes the maximum input sequence length for LLM.

#### Testing Wikitext2

```
bash scripts/run_wikitext2.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--device` : Denotes which device to place the model onto.
- `--seqlen` : Denotes the maximum input sequence length for the model.

#### Testing advGLUE

```
bash scripts/run_advglue.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--data_file` : Dataset file location.
- `--ntrain` : number of shots.
- `--test_origin` : Denotes whether to test on the original GLUE data.

#### Testing TruthfulQA

```
bash scripts/run_tqa.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--presets` : Preset to use for prompt generation. Please see tqa_presets.py for options.
- `--input_path` : Dataset file location.
- `--device` : Denotes which device to place the model onto.

#### Testing FLOPs (floating point operations)

```
bash scripts/run_flops.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--seqlen` : Denotes the input sequence length for the model.

## Acknowledgements

In addition to the code in this repo, we also use [EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models. (github.com)](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation.

## Citation

```
@inproceedings{yang2024llmcbench,
  title={LLMCBench: Benchmarking Large Language Model Compression for Efficient Deployment},
  author={Yang, Ge and He, Changyi and Guo, Jinyang and Wu, Jianyu and Ding, Yifu and Liu, Aishan and Qin, Haotong and Ji, Pengliang and Liu, Xianglong},
  booktitle={Thirty-eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2024}
}
```
