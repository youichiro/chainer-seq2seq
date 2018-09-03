#!/usr/bin/env python
#-*- coding: utf-8 -*-

def model_opts(parser):
    # Embedding options
    group = parser.add_argument_group('Model-Embeddings')

    # Encoder-Decoder options
    group = parser.add_argument_group('Model-Encoder-Decoder')
    group.add_argument('--layers', type=int, default=2, 
        help='number of rnn layers')
    group.add_argument('--units', type=int, default=500,
        help='number of hidden units of rnn')
    group.add_argument('--rnn', default='LSTM', choices=['LSTM'], 
        help='type of recurrent neural network')
    group.add_argument('--reverse_encoding', action='store_true',
        help='Whether RNN reads a input sentence in reverse')

   # Attentional mechanism options
    group = parser.add_argument_group('Model-Attentional mechanism')
    group.add_argument('--attn', default='disuse',
        choices=['disuse', 'local', 'global'],
        help='''type of attentional mechanism  
                diuse: model unused attentional mechanism
                local: local attention model 
                global: global attention model''')
    group.add_argument('--score', default='dot',
        choices=['dot', 'general', 'concat'],
        help='''scoring method in attentional mechanism
              please refer to "Effective Approaches to Attention-based Neural 
              Machine Translation; Proceedings of the 2015 Conference on 
              Empirical Methods in Natural Language Processing"''')
    group.add_argument('--input-feeding', action='store_true',
        help='Whether attentional vectors are fed as input to the next steps')


def train_opts(parser):
    # General options
    group = parser.add_argument_group('General')
    group.add_argument('--save-dir')
    group.add_argument('--model', default='model')
    group.add_argument('--gpuid', type=int, default=-1)

    # Data options
    group = parser.add_argument_group('Data')
    group.add_argument('--train-src',
                        help='source sentence list for training')
    group.add_argument('--train-tgt',
                        help='target sentence list for training')
    group.add_argument('--valid-src',
                        help='source sentence list for validation')
    group.add_argument('--valid-tgt', 
                        help='target sentence list for validation')

    # Vocabulary options
    group = parser.add_argument_group('Vocab')
    group.add_argument('--src-vocabsize', type=int, default=50000)
    group.add_argument('--tgt-vocabsize', type=int, default=50000)
    group.add_argument('--src-minfreq', type=int, default=1)
    group.add_argument('--tgt-minfreq', type=int, default=1)

    # Truncation options
    group = parser.add_argument_group('Pruning')
    group.add_argument('--src-minlen', type=int, default=4)
    group.add_argument('--tgt-minlen', type=int, default=4)
    group.add_argument('--src-maxlen', type=int, default=70)
    group.add_argument('--tgt-maxlen', type=int, default=70)

    # Optimization options
    group = parser.add_argument_group('Optimization')
    group.add_argument('--batchsize', type=int, default=128)
    group.add_argument('--epochs', type=int, default=20)
    group.add_argument('--dout', type=float, default=0.3)
    group.add_argument('--optim', default='Adam', choices=['SGD', 'Adam'])

    # Validation options
    group = parser.add_argument_group('Validation')
    group.add_argument('--report-sbleu', action='store_true')
    group.add_argument('--report-sari', action='store_true')
    group.add_argument('--maxlen', type=int, default=70)

    # Logging options
    group = parser.add_argument_group('Logging')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='number of iteration to show log')


def translate_opts(parser):
    # General options
    group = parser.add_argument_group('General')
    group.add_argument('--model', help='model for translation')
    group.add_argument('--src', help='source sentence list for translation')
    group.add_argument('--tgt', help='reference sentence list')
    group.add_argument('--output', help='file name for saving model output')
    group.add_argument('--gpuid', type=int, default=-1)
    group.add_argument('--maxlen', default=70, help='maximum length of output sentence')
    group.add_argument('--batchsize', type=int, default=100)

    # Beam search options
    group = parser.add_argument_group('Beam search')
    group.add_argument('--beamsize', type=int, default=1, help='beam size')
    group.add_argument('--n_cands', type=int, default=1, 
                       help='number of output candidates')
    group.add_argument('--ranking',  default='None',
                       choices=['None', 'sbleu', 'sari'],
                       help='ranking method')


def evaluation_opts(parser):
    # General options
    group = parser.add_argument_group('General')
    group.add_argument('--src', default='./test_data/kyoto-train.ja',
                        help='source sentence list for translation')
    group.add_argument('--tgt', default='./pred.txt',
                        help='reference sentence list')
    group.add_argument('--eval_sbleu', action='store_true')
    group.add_argument('--eval_sari', action='store_true')

