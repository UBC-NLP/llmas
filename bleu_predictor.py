'''
predict bleu score via linear regression model
'''

import random, argparse
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from scipy import stats
from tqdm import trange

class Net(nn.Module):
  def __init__(self, feature_dim, hidden_dim, hidden_layer_num):
    super(Net, self).__init__()
    self.first_layer = nn.Linear(feature_dim, hidden_dim)
    self.layers = nn.ModuleList()
    for i in range(hidden_layer_num):
      self.layers.append(nn.Linear(hidden_dim, hidden_dim))
    self.predict = nn.Linear(hidden_dim, 1)

  def forward(self, x):
    x = F.relu(self.first_layer(x))
    for i in range(len(self.layers)):
      x = F.relu(self.layers[i](x))
    x = self.predict(x)
    return x

def convert_gene_to_arch_info(gene):
  arch_info = {}
  arch_info["encoder-embed-dim-subtransformer"] = gene["encoder"]["encoder_embed_dim"]
  arch_info["encoder-layer-num-subtransformer"] = gene["encoder"]["encoder_layer_num"]
  arch_info["encoder-ffn-embed-dim-all-subtransformer"] = gene["encoder"]["encoder_ffn_embed_dim"]
  arch_info["encoder-self-attention-heads-all-subtransformer"] = gene["encoder"]["encoder_self_attention_heads"]
  arch_info["decoder-embed-dim-subtransformer"] = gene["decoder"]["decoder_embed_dim"]
  arch_info["decoder-layer-num-subtransformer"] = gene["decoder"]["decoder_layer_num"]
  arch_info["decoder-ffn-embed-dim-all-subtransformer"] = gene["decoder"]["decoder_ffn_embed_dim"]
  arch_info["decoder-self-attention-heads-all-subtransformer"] = gene["decoder"]["decoder_self_attention_heads"]
  arch_info["decoder-ende-attention-heads-all-subtransformer"] = gene["decoder"]["decoder_ende_attention_heads"]
  arch_info["decoder-arbitrary-ende-attn-all-subtransformer"] = gene["decoder"]["decoder_arbitrary_ende_attn"]
  return arch_info

def convert_arch_info_to_features(arch_info, feature_type="hat"):
  new_arch_info = {}
  for info in arch_info:
    if ":" in info:
      new_arch_info[info.split(":")[0]] = eval(info.split(":")[1])
    else:
      new_arch_info[info] = arch_info[info]
  arch_info = new_arch_info
  features = []
  if feature_type == "hat":
    # hat's [640, 6, 2048, 6, 640, 6, 2048, 6, 6, 2]
    # ours [640.0, 6.0, 3072.0, 8.0, 640.0, 6.0, 3072.0, 8.0, 8.0, 3.0]
    features.append(arch_info["encoder-embed-dim-subtransformer"]/640.0)
    features.append(arch_info["encoder-layer-num-subtransformer"]/6.0)
    features.append(np.mean(arch_info["encoder-ffn-embed-dim-all-subtransformer"][0:arch_info["encoder-layer-num-subtransformer"]])/3072.0)
    features.append(np.mean(arch_info["encoder-self-attention-heads-all-subtransformer"][0:arch_info["encoder-layer-num-subtransformer"]])/8.0)
    features.append(arch_info["decoder-embed-dim-subtransformer"]/640.0)
    features.append(arch_info["decoder-layer-num-subtransformer"]/6.0)
    features.append(np.mean(arch_info["decoder-ffn-embed-dim-all-subtransformer"][0:arch_info["decoder-layer-num-subtransformer"]])/3072.0)
    features.append(np.mean(arch_info["decoder-self-attention-heads-all-subtransformer"][0:arch_info["decoder-layer-num-subtransformer"]])/8.0)
    features.append((np.mean(arch_info["decoder-ende-attention-heads-all-subtransformer"][0:arch_info["decoder-layer-num-subtransformer"]]))/8.0)
    features.append((1.0+np.mean(arch_info["decoder-arbitrary-ende-attn-all-subtransformer"][0:arch_info["decoder-layer-num-subtransformer"]]))/3.0)
  elif feature_type == "fine":
    features.append(arch_info["encoder-embed-dim-subtransformer"]/640.0)
    features.append(arch_info["encoder-layer-num-subtransformer"]/6.0)
    for lay_idx in range(6):
      if lay_idx < arch_info["encoder-layer-num-subtransformer"]:
        features.append(arch_info["encoder-ffn-embed-dim-all-subtransformer"][lay_idx])
      else:
        features.append(0)
    for lay_idx in range(6):
      if lay_idx < arch_info["encoder-layer-num-subtransformer"]:
        features.append(arch_info["encoder-self-attention-heads-all-subtransformer"][lay_idx])
      else:
        features.append(0)
    features.append(arch_info["decoder-embed-dim-subtransformer"]/640.0)
    features.append(arch_info["decoder-layer-num-subtransformer"]/6.0)
    for lay_idx in range(6):
      if lay_idx < arch_info["decoder-layer-num-subtransformer"]:
        features.append(arch_info["decoder-ffn-embed-dim-all-subtransformer"][lay_idx])
      else:
        features.append(0)
    for lay_idx in range(6):
      if lay_idx < arch_info["decoder-layer-num-subtransformer"]:
        features.append(arch_info["decoder-self-attention-heads-all-subtransformer"][lay_idx])
      else:
        features.append(0)
    for lay_idx in range(6):
      if lay_idx < arch_info["decoder-layer-num-subtransformer"]:
        features.append(arch_info["decoder-ende-attention-heads-all-subtransformer"][lay_idx])
      else:
        features.append(0)
    for lay_idx in range(6):
      if lay_idx < arch_info["decoder-layer-num-subtransformer"]:
        features.append(arch_info["decoder-arbitrary-ende-attn-all-subtransformer"][lay_idx])
      else:
        features.append(0)

  return features

class BleuPredictor(object):
  def __init__(self, x_train, y_train_teacher, x_test, y_test_gold, feature_dim, hidden_dim, hidden_layer_num, train_steps, bsz, lr, save_ckpt, save_file):
    self.x_train = x_train
    self.y_train_teacher = y_train_teacher
    self.x_test = x_test
    self.y_test_gold = y_test_gold
    self.train_steps = train_steps
    self.bsz = bsz
    self.feature_dim = feature_dim
    self.hidden_dim = hidden_dim
    self.hidden_layer_num = hidden_layer_num
    self.lr = lr
    self.feature_norm = [640.0, 6.0, 3072.0, 8.0, 640.0, 6.0, 3072.0, 8.0, 8.0, 3.0]
    self.save_ckpt = save_ckpt
    self.save_file = save_file

    self.model = Net(self.feature_dim, self.hidden_dim, self.hidden_layer_num)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    self.criterion = torch.nn.MSELoss()
    if torch.cuda.is_available():
      self.model = self.model.to(0)
      self.criterion = self.criterion.to(0)

  def train(self):
    for i in trange(self.train_steps):
      sample_ind = random.sample(range(len(self.x_train)), k=self.bsz)
      sample_x = [self.x_train[sample_ind[k]] for k in range(self.bsz)]
      sample_y = [self.y_train_teacher[sample_ind[k]] for k in range(self.bsz)]

      sample_x_tensor = torch.Tensor(sample_x)
      sample_y_tensor = torch.Tensor(sample_y)
      if torch.cuda.is_available():
        sample_x_tensor = sample_x_tensor.to(0)
        sample_y_tensor = sample_y_tensor.to(0)

      prediction = self.model(sample_x_tensor).squeeze()

      loss = self.criterion(prediction, sample_y_tensor)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
    if self.save_ckpt == 1:
      torch.save(self.model.state_dict(), self.save_file)

  def load_ckpt(self, ckpt_path):
    self.model.load_state_dict(torch.load(ckpt_path))

  def predict_bleu(self, config):
    with torch.no_grad():
      features = convert_arch_info_to_features(convert_gene_to_arch_info(config))
      features = torch.Tensor(features) # np.array(features))
      if torch.cuda.is_available():
        features = features.to(0)
      prediction = self.model(features).cpu().item()
    return prediction

  def test(self):
    abs_diff, n = 0.0, 0.0
    kendal = [[], []]
    with torch.no_grad():
      sample_x_tensor = torch.Tensor(self.x_test)
      sample_y_tensor = torch.Tensor(self.y_test_gold)
      if torch.cuda.is_available():
        sample_x_tensor = sample_x_tensor.to(0)
        sample_y_tensor = sample_y_tensor.to(0)
      prediction = self.model(sample_x_tensor).squeeze()
      for cur_pred, cur_gold in zip(prediction, y_test_gold):
        abs_diff += abs(cur_pred.cpu().item()-cur_gold)
        n += 1.0
        kendal[0].append(cur_pred.cpu().item())
        kendal[1].append(cur_gold)
    mae = abs_diff/n
    ktau = stats.kendalltau(kendal[0], kendal[1])[0]
    return mae, ktau

if __name__=='__main__':
  parser = argparse.ArgumentParser(description="bleu predictor")
  parser.add_argument('--manual_seed', type=int, default=123, help='manual seed')
  parser.add_argument("--gpt_scorer_outputs", type=str, default="/Users/ganeshj/Desktop/ubc_proj/hatv2/slurm/experiments/gpt_scorer_outputs", help="folder for gpt scorer outputs")
  parser.add_argument("--testset_outputs", type=str, default="/Users/ganeshj/Desktop/ubc_proj/hatv2/slurm/experiments/train-test_seedarchs", help="folder for testset outputs")
  parser.add_argument("--task", type=str, default="wmt14ende", help="folder for gpt scorer outputs")
  parser.add_argument("--teacher_model", type=str, default="gpt-35-turbo", help="bleu generator model")
  parser.add_argument('--feature-dim', type=int, default=10, help='dimension of feature vector')
  parser.add_argument('--hidden-dim', type=int, default=400, help='hidden dimension of FC layers in bleu predictor')
  parser.add_argument('--hidden-layer-num', type=int, default=3, help='number of FC layers')
  parser.add_argument('--bsz', type=int, default=128, help='bleu predictor training batch size')
  parser.add_argument('--lr', type=float, default=1e-5, help='bleu predictor training learning rate')
  parser.add_argument('--train-steps', type=int, default=5000, help='bleu predictor training steps')
  parser.add_argument("--feature_type", type=str, default="hat", help="hat or fine")
  parser.add_argument('--src_seeds', type=int, nargs='+', help='seeds', default=[123, 456, 789])
  parser.add_argument('--save-ckpt', type=int, default=0, help='1 for save, 0 for dont save')
  parser.add_argument('--save-file', type=str, default="/tmp/model.ckpt", help='full path for checkpoint to be saved')
  args = parser.parse_args()

  random.seed(args.manual_seed)
  np.random.seed(args.manual_seed)
  torch.manual_seed(args.manual_seed)
  torch.cuda.manual_seed_all(args.manual_seed)

  maes, kendals = [], []
  for src_seed in args.src_seeds:
    # read training set
    x_train, y_train_teacher = [], []
    for dest_seed in [111, 222, 333, 444, 555, 666]:
      for line in open(args.gpt_scorer_outputs + "/june13-gendata-%d-%s"%(dest_seed, args.teacher_model) + "/generations.jsonl"):
        content = json.loads(line.strip())
        if content["seed"] == src_seed and content["dataset"] == args.task and content["openai_model"] == args.teacher_model:
          x_train.append(convert_arch_info_to_features(content["scratch"]["arch_info"], args.feature_type))
          y_train_teacher.append(content["oai_valid_BLEU"])
    # x_train = x_train[0:10]
    # y_train_teacher = y_train_teacher[0:10]
    # x_train = np.array(x_train)
    # y_train_teacher = np.array(y_train_teacher)
    # read test set
    x_test, y_test_gold = [], []
    for line in open(args.testset_outputs + "/" + str(src_seed) + "/" + args.task + "/test.jsonl"):
      content = json.loads(line.strip())
      y_test_gold.append(content["scratch"]["valid_BLEU"])
      x_test.append(convert_arch_info_to_features(content["scratch"]["arch_info"], args.feature_type))
      if len(x_test) == 1:
        args.feature_dim = len(convert_arch_info_to_features(content["scratch"]["arch_info"], args.feature_type))
    # x_test = np.array(x_test)
    # y_test_gold = np.array(y_test_gold)
    print("#train = %d"%(len(x_train)))
    print("#test = %d"%(len(x_test)))
    print("feature-dim = %d"%(args.feature_dim))
    bleu_predict_model = BleuPredictor(x_train, y_train_teacher, x_test, y_test_gold, args.feature_dim, args.hidden_dim, args.hidden_layer_num, args.train_steps, args.bsz, args.lr, args.save_ckpt, args.save_file)
    bleu_predict_model.train()
    mae, kendal = bleu_predict_model.test()
    maes.append(mae)
    kendals.append(kendal)
  print("%.2f (%.2f),%.2f (%.2f)"%(np.mean(maes), np.std(maes), np.mean(kendals), np.std(kendals)))



