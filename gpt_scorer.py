'''
code to generate performance score for each architecture
'''

from scipy import stats
import numpy as np
import sys, os, glob, random, json, time, argparse, openai, traceback, uuid, subprocess, copy
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

parser = argparse.ArgumentParser(description="script to generate performance estimates")
parser.add_argument("--source_dir", type=str, default="/mnt/data1/ganesh/experiments/hatv2/train-test_seedarchs", help="source directory for data")
parser.add_argument("--dest_dir", type=str, default="/mnt/data1/ganesh/experiments/hatv2/gpt_scorer_outputs", help="destination directory for outputs")
parser.add_argument("--template_dir", type=str, default="/mnt/data1/ganesh/experiments/hatv2/instruction_templates", help="directory for templates")
parser.add_argument("--experiment_name", type=str, default="june9_wmt14ende_chatgpt_no-instruction", help="name for the experiment")
parser.add_argument("--datasets", nargs='+', default=["wmt14ende", "wmt14enfr", "wmt19ende"], help="mt datasets")
parser.add_argument("--seeds", nargs='+', default=[123, 456, 789], help="seed directories")
parser.add_argument("--method_name", type=str, default="gpt-35-turbo_no-instruction", help="method name")
parser.add_argument("--prompt_template_f", type=str, default="gpt_scorer_no_instruction", help="template file")
parser.add_argument("--openai_key", type=str, default="apr22_1", help="id for openai key")
parser.add_argument("--openai_models", nargs='+', default=["gpt-35-turbo", "gpt-4"], help="id for openai models")
parser.add_argument("--datasets_id", type=str, default="wmt16", help="id of the dataset in the datasets library")
parser.add_argument("--datasets_subset", type=str, default="de-en", help="subset of the dataset")
parser.add_argument("--datasets_split", type=str, default="train", help="split of the dataset")
parser.add_argument("--datasets_source_field", type=str, default="en", help="source field of the datasets")
parser.add_argument("--datasets_target_field", type=str, default="de", help="target field of the datasets")
parser.add_argument("--num_task_examples", type=int, default=5, help="no. of downstream examples to add to the instruction")
parser.add_argument("--num_train_examples", type=int, default=10, help="no. of seed architecture examples to add to the instruction")

# create the openai client
def key_factory(key):
    return None

def extract_bleu_v1(response):
  response = response.strip()
  if len(response) == 0:
    return None
  response = response.split()[0]
  try:
    float(response)
    return float(response)
  except ValueError:
    return None


def main():
    args = parser.parse_args()

    cur_exp_dir = args.dest_dir + "/" + args.experiment_name
    os.makedirs(cur_exp_dir, exist_ok=True)
  
    access_info = key_factory(args.openai_key)
    openai.api_type = access_info["api_type"]
    openai.api_base = access_info["api_base"]
    openai.api_version = access_info["api_version"]
    openai.api_key = access_info["api_key"]

    w = open(cur_exp_dir + "/generations.jsonl", "w")
    for openai_model in args.openai_models:
      res = openai_model
      for dataset in args.datasets:
        maes, kendals = [], []
        for seed in args.seeds:
          random.seed(seed)
          # cur_seed_dir = cur_exp_dir + "/" + dataset + "/" + str(seed)
          # os.makedirs(cur_seed_dir, exist_ok=True)

          # read train, test examples
          cur_src_dir = args.source_dir + "/" + str(seed) + "/" + dataset
          train_examples, test_examples = [], []
          for line in open(cur_src_dir + "/train.jsonl"):
            train_examples.append(json.loads(line.strip()))
            if len(train_examples) == args.num_train_examples:
              break
          print("# train examples = %d"%len(train_examples))
          for line in open(cur_src_dir + "/test.jsonl"):
            test_examples.append(json.loads(line.strip()))
          
          # sample task examples
          sampled_examples = {}
          task_dataset = load_dataset(args.datasets_id, args.datasets_subset, split=args.datasets_split)
          print("# task-specific examples: %d"%(len(task_dataset)))
          while len(sampled_examples) < args.num_task_examples:
            rand_idx = random.randint(0, len(task_dataset)-1)
            if rand_idx in sampled_examples:
                continue
            sampled_examples[rand_idx] = {"src": task_dataset[rand_idx]['translation'][args.datasets_source_field], "trg": task_dataset[rand_idx]['translation'][args.datasets_target_field]}

          task_examples_str = ""
          i = 0
          for sample_id in sorted(sampled_examples):
              task_examples_str += "Example %d:\n"%(i+1)
              task_examples_str += "Input: %s\n"%(sampled_examples[sample_id]["src"])
              task_examples_str += "Output: %s\n\n"%(sampled_examples[sample_id]["trg"])
              i += 1
          task_examples_str = task_examples_str.strip()
            
          # sample seed architecture examples
          seed_archs_str = ""
          for i, arch in enumerate(train_examples):
              # seed_archs_str += "Example %d:\n"%(i+1)
              seed_archs_str += "\n".join(arch["scratch"]["arch_info"]) + "\n"
              seed_archs_str += "GFLOPS: %.1f"%(arch["scratch"]["total_flops"]/1000000000.0) + "\n"
              seed_archs_str += "BLEU: %.2f"%(arch["scratch"]["valid_BLEU"]) + "\n\n"
          seed_archs_str = seed_archs_str.strip()

          # prepare the prompt
          prompt_template = ""
          for line in open(args.template_dir + "/" + dataset + "/" + args.prompt_template_f):
              prompt_template += line
          prompt_content_str = prompt_template
          for match_str, replace_str in [("$$$task_examples$$$", task_examples_str), ("$$$seedarch_examples$$$", seed_archs_str)]:
              prompt_content_str = prompt_content_str.replace(match_str, replace_str)
          
          abs_diff, n = 0.0, 0.0
          kendal = [[], []]
          pbar = tqdm(total=len(test_examples))
          for test_example in test_examples:
            cur_prompt = copy.deepcopy(prompt_content_str)
            cur_prompt += "\n\n"
            # cur_prompt += "Final  %d:\n"%(len(train_examples)+1)
            # cur_prompt += "Architecture for :"
            cur_prompt += "\n".join(test_example["scratch"]["arch_info"]) + "\n"
            cur_prompt += "GFLOPS: %.1f"%(test_example["scratch"]["total_flops"]/1000000000.0) + "\n"
            cur_prompt += "BLEU: "
            # messages = [{"role": "system", "content": "You are a performance estimator for machine translation task, where you will  predict the BLEU score for the test architecture."}, {"role": "user", "content": cur_prompt}]
            messages = [{"role": "user", "content": cur_prompt}]
            is_done = False
            cur_bleu_score = None
            while is_done == False:
              time.sleep(1)
              try:
                response_dict = openai.ChatCompletion.create(engine=openai_model, messages=messages, max_tokens=25)
                response_str = response_dict["choices"][0]["message"]['content']
                cur_bleu_score = extract_bleu_v1(response_str)
                # print(cur_bleu_score, response_str)
                if cur_bleu_score is not None:
                  is_done = True
              except:
                # traceback.print_exc()
                continue
            n += 1.0
            if "valid_BLEU" in test_example["scratch"]:
              abs_diff += abs(cur_bleu_score-test_example["scratch"]["valid_BLEU"])
              kendal[0].append(cur_bleu_score)
              kendal[1].append(test_example["scratch"]["valid_BLEU"])
            w.write(json.dumps({"seed": seed, "openai_model": openai_model, "dataset": dataset, "scratch": test_example["scratch"], "oai_valid_BLEU": cur_bleu_score})+"\n")
            pbar.update(1)
          pbar.close()
          if len(kendal[0]) > 0:
            maes.append(abs_diff/n)
            kendals.append(stats.kendalltau(kendal[0], kendal[1])[0])
        if len(maes) > 0:
          res += ",%.2f (%.2f),%.2f (%.2f)"%(np.mean(maes), np.std(maes), np.mean(kendals), np.std(kendals))
      print(res)
    w.close()

main()


