import argparse
import pickle
import os
from chainer import serializers
from chainer import cuda
from Utils import word2id
from Opts import translate_opts
from Nets import Seq2seq
from train import encdec_convert

def parse_args():
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    translate_opts(parser)
    opts = parser.parse_args()
    return opts


def main():
    opts = vars(parse_args())
    
    # Load model's parameters and a vocabulary
    prefix = os.path.join(os.path.dirname(opts['model']), 
                          os.path.basename(opts['model']).split('-')[0])
    params = pickle.load(open('{}.opts'.format(prefix), 'br'))
    svocab = [w.rstrip() for w in open('{}.svocab'.format(prefix), 'r')]
    tvocab = [w.rstrip() for w in open('{}.tvocab'.format(prefix), 'r')]
    
    # Setup models
    svocab_size = len(svocab) + 3 # 3 means number of special tags such as
    tvocab_size = len(tvocab) + 3 # "UNK", "BOS", and "EOS" 
    model = Seq2seq(svocab_size, tvocab_size, params)
    serializers.load_npz(opts['model'], model)
    if opts['gpuid'] >= 0:
        cuda.get_device(opts['gpuid']).use()
        model.to_gpu(opts['gpuid'])

    # Setup a data
    test_src = word2id(opts['src'], svocab)
    test_tgt = word2id(opts['tgt'], tvocab)
    test_data = [(s, t) for s, t in zip(test_src, test_tgt)]

    # Translating
    id2word = ['UNK', 'BOS', 'EOS'] + tvocab
    if opts['beamsize'] < 2:
        for i in range(0, len(test_data), opts['batchsize']):
            srcs, _, refs = encdec_convert(
                                test_data[i:i+opts['batchsize']], opts['gpuid'])
            hyps = model.translate(srcs, opts['maxlen'])
            for hyp in hyps:
                out = ' '.join([id2word[i] for i in hyp])
                print(out)
    else:
        for seqs in test_data:
            src, _, ref = encdec_convert([seqs], opts['gpuid'])
            hyps = model.beam(src, ref, opts['maxlen'], opts['beamsize'], 
                              opts['n_cands'], opts['ranking'])
            for hyp in hyps:
                out = ' '.join([id2word[i] for i in hyp])
                print(out, end='\t')
            print('')


if __name__ == '__main__':
    main()
