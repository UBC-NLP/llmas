"""
BLEU scoring of generated translations against reference translations.
"""

import argparse
import os
import sys

from fairseq import bleu
from fairseq.data import dictionary


def get_parser():
    parser = argparse.ArgumentParser(description='Command-line script for BLEU scoring.')
    # fmt: off
    parser.add_argument('-s', '--sys', default='-', help='system output')
    parser.add_argument('-r', '--ref', required=True, help='references')
    parser.add_argument('-o', '--order', default=4, metavar='N',
                        type=int, help='consider ngrams up to this order')
    parser.add_argument('--ignore-case', action='store_true',
                        help='case-insensitive scoring')
    parser.add_argument('--sacrebleu', action='store_true',
                        help='score with sacrebleu')
    parser.add_argument('--sentence-bleu', action='store_true',
                        help='report sentence-level BLEUs (i.e., with +1 smoothing)')
    # fmt: on
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    assert args.sys == '-' or os.path.exists(args.sys), \
        "System output file {} does not exist".format(args.sys)
    assert os.path.exists(args.ref), \
        "Reference file {} does not exist".format(args.ref)

    dict = dictionary.Dictionary()

    def readlines(fd):
        for line in fd.readlines():
            if args.ignore_case:
                yield line.lower()
            else:
                yield line

    if args.sacrebleu:
        import sacrebleu

        def score(fdsys):
            #with open(args.ref) as fdref:
            #    print(sacrebleu.corpus_bleu(fdsys, [fdref]))
            refs = []
            for line in open(args.ref):
                refs.append(line.strip())
            syss = []
            for line in open(args.sys):
                syss.append(line.strip())
            print(sacrebleu.corpus_bleu(syss, [refs]))
            # from torchmetrics import SacreBLEUScore
            # sacre_bleu = SacreBLEUScore()
            # print("SacreBLEU: %.2f"%(100.0*sacre_bleu(syss, refs).item()))

    elif args.sentence_bleu:
        def score(fdsys):
            with open(args.ref) as fdref:
                scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())
                for i, (sys_tok, ref_tok) in enumerate(zip(readlines(fdsys), readlines(fdref))):
                    scorer.reset(one_init=True)
                    sys_tok = dict.encode_line(sys_tok)
                    ref_tok = dict.encode_line(ref_tok)
                    scorer.add(ref_tok, sys_tok)
                    print(i, scorer.result_string(args.order))
    else:
        print('this scorer')
        def score(fdsys):
            with open(args.ref) as fdref:
                scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())
                for sys_tok, ref_tok in zip(readlines(fdsys), readlines(fdref)):
                    sys_tok = dict.encode_line(sys_tok)
                    ref_tok = dict.encode_line(ref_tok)
                    scorer.add(ref_tok, sys_tok)
                print(scorer.result_string(args.order))

    if args.sys == '-':
        score(sys.stdin)
    else:
        with open(args.sys, 'r') as f:
            score(f)
        # score(None)


if __name__ == '__main__':
    main()
