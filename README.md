# LLM Performance Predictors are good initializers for Architecture Search

This repository contains the code and the data used in the [LLM-PP work](https://arxiv.org/abs/2310.16712). This repository builds on [Hardware Aware Transformer (HAT)'s repository](https://github.com/mit-han-lab/hardware-aware-transformers).

## Data ([data/](data/))
1. [Train from scratch Architectures (30)](data/train-test_seedarchs)
2. [PP Prompt Instruction templates](data/instruction_templates)
3. [LLM PP Data Generator Input](data/bleu_predictor_data)

## LLM-PP Data Generator

To generate LLM-PP GPT-4 predictions for WMT'14 En-De, run this:

```
python gpt_scorer.py --source_dir data/train-test_seedarchs --dest_dir /tmp/gpt_scorer_outputs --template_dir data/instruction_templates --experiment_name nov15_exp --datasets wmt14ende --prompt_template_f gpt_scorer_concentrate_statement --openai_models gpt-4
```

## LLM-Distill-PP Trainer

To train LLM-Distill-PP model, run this:

```
python bleu_predictor.py --gpt_scorer_outputs /tmp/gpt_scorer_outputs --testset_outputs data/train-test_seedarchs --task wmt14ende --teacher_model gpt-4 --save-file /tmp/model.ckpt
```

## Hybrid-search

To execute hybrid-search NAS algorithm, run this:

```
CUDA_VISIBLE_DEVICES=0 python evo_search.py --configs=configs/wmt14.en-de/supertransformer/space0.yml --evo-configs=configs/wmt14.en-de/evo_search/wmt14ende_titanxp.yml --evo-iter 30 --population-size 125 --parent-size 25 --mutation-size 50 --crossover-size 50 --mutation-prob 0.3 --latency-constraint 150 --bleu-ckpt-path  /tmp/model.ckpt --bleu-predictor-start-idx 0 --bleu-predictor-end-idx 14
```

## Citation
If you use this code, please cite:
```
@inproceedings{jawahar2024llmpp,
      title={LLM Performance Predictors are good initializers for Architecture Search}, 
      author={Ganesh Jawahar and Muhammad Abdul-Mageed and Laks V. S. Lakshmanan and Dujian Ding},
      year={2024},
      booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
}
```

### License
This repository is MIT-licensed.

