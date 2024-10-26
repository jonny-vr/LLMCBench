# LLMCBench: Benchmarking Large Language Model Compression for Efficient Deployment

![image-20241026195404186](C:\Users\一觞浮云醉月光i\AppData\Roaming\Typora\typora-user-images\image-20241026195404186.png)

## Installation

```
git clone https://github.com/AboveParadise/LLMCBench.git
cd LLMCBench

conda create -n llmcbench python=3.9
conda activate llmcbench

pip install transformers
```

​    

## Example

```
bash script/run_mmlu.sh
```

​    

### Evaluation

In addition to the code in this repo, we also use [EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models. (github.com)](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation.

## Citation
