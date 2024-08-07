'''
100ms
python other_bleu_metrics.py /scratch/st-amuham01-1/ganeshjw/objects/hatv2/nov24_nas_hat_arch_neuron_difflatency/nov24_nas_hat_wmt14.en-de/hat_100/perf_scratch_test
python other_bleu_metrics.py /scratch/st-amuham01-1/ganeshjw/objects/hatv2/nov24_nas_hat_arch_neuron_difflatency/nov24_nas_archexp_wmt14.en-de_archrouting_jack_drop_2L/arch_100/perf_scratch_test
python other_bleu_metrics.py /scratch/st-amuham01-1/ganeshjw/objects/hatv2/nov24_nas_hat_arch_neuron_difflatency/nov24_nas_neuronexp_wmt14.en-de_neuronrouting_jack_drop_2L/neur_100/perf_scratch_test
python other_bleu_metrics.py /scratch/st-amuham01-1/ganeshjw/objects/nas-moe/june30_bleu-predictor_hat_latency-constraint_100_15000_gpt-4_wmt14ende_hat_hybrid_0_14/baseline/perf_scratch_test

150ms
python other_bleu_metrics.py /scratch/st-amuham01-1/ganeshjw/objects/hatv2/nov24_nas_hat_arch_neuron_difflatency/nov24_nas_hat_wmt14.en-de/hat_150/perf_scratch_test
python other_bleu_metrics.py /scratch/st-amuham01-1/ganeshjw/objects/hatv2/nov24_nas_hat_arch_neuron_difflatency/nov24_nas_archexp_wmt14.en-de_archrouting_jack_drop_2L/arch_150/perf_scratch_test
python other_bleu_metrics.py /scratch/st-amuham01-1/ganeshjw/objects/hatv2/nov24_nas_hat_arch_neuron_difflatency/nov24_nas_neuronexp_wmt14.en-de_neuronrouting_jack_drop_2L/neur_150/perf_scratch_test
python other_bleu_metrics.py /scratch/st-amuham01-1/ganeshjw/objects/nas-moe/aug19_bleu-predictor_wmt14ende_150-200_latency-constraint_150_15000_gpt-4_wmt14ende_hat_hybrid_0_14/baseline/perf_scratch_test

200ms
python other_bleu_metrics.py /scratch/st-amuham01-1/ganeshjw/objects/hatv2/nov24_nas_hat_arch_neuron_difflatency/nov24_nas_hat_wmt14.en-de/hat_200/perf_scratch_test
python other_bleu_metrics.py /scratch/st-amuham01-1/ganeshjw/objects/hatv2/nov24_nas_hat_arch_neuron_difflatency/nov24_nas_archexp_wmt14.en-de_archrouting_jack_drop_2L/arch_200/perf_scratch_test
python other_bleu_metrics.py /scratch/st-amuham01-1/ganeshjw/objects/hatv2/nov24_nas_hat_arch_neuron_difflatency/nov24_nas_neuronexp_wmt14.en-de_neuronrouting_jack_drop_2L/neur_200/perf_scratch_test
python other_bleu_metrics.py /scratch/st-amuham01-1/ganeshjw/objects/nas-moe/aug19_bleu-predictor_wmt14ende_150-200_latency-constraint_200_15000_gpt-4_wmt14ende_hat_hybrid_0_14/baseline/perf_scratch_test

wmt'14 en-fr
python other_bleu_metrics.py /scratch/st-amuham01-1/ganeshjw/objects/hatv2/nov24_nas_hat_arch_neuron_difflatency/nov24_nas_hat_wmt19.en-de/hat_200/perf_scratch_test

'''


import sys, os, glob
import sacrebleu
from sacremoses import MosesTokenizer, MosesDetokenizer
mt = MosesDetokenizer(lang='de')

folder = sys.argv[1]
sys_f = glob.glob(folder + "/*test*.sys")[0]
ref_f = glob.glob(folder + "/*test*.ref")[0]
refs = []
for line in open(ref_f):
  refs.append((line.strip()))
syss = []
for line in open(sys_f):
  syss.append(mt.detokenize(line.strip()))
print("sacrebleu", sacrebleu.corpus_bleu(syss, [refs]))

'''
import datasets
for name in ["sacrebleu", "bleurt"]:
  metric = datasets.load_metric(name)
  print(metric.compute(predictions=syss, references=refs))
'''


