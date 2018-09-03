import argparse
import pickle
import os
from bleu import compute_bleu
import chainer
from chainer import training
from chainer.training import extensions
from chainer.backends import cuda
from chainer.dataset import convert
from chainer import serializers
from Utils import get_vocab, save_vocab, word2id
from Nets import Seq2seq
from Opts import train_opts, model_opts
from Const import UNK_ID, BOS_ID, EOS_ID, IGNORE



class SaveModel(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, model, save_dir, name):
        self.model = model
        self.save_dir = save_dir
        self.name = name

    def __call__(self, trainer):
        model_name = '{}-e{}.model'.format(self.name, str(trainer.updater.epoch))
        save_path = os.path.join(self.save_dir, model_name)
        serializers.save_npz(save_path, self.model)


class CalculateSBLEU(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, model, test_data, key, batch, device, maxlen):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.maxlen = maxlen

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            refs = []
            hyps = []
            for i in range(0, len(self.test_data), self.batch):
                srcs, tgts = zip(*self.test_data[i:i+self.batch])
                refs.extend([[t.tolist()] for t in tgts])
                
                srcs = [chainer.dataset.to_device(self.device, x) for x in srcs]
                oys = self.model.translate(srcs, self.maxlen)
                hyps.extend(oys)
        sbleu = [compute_bleu(ref, hyp, smooth=True) for ref, hyp in zip(refs, hyps)]
        sbleu = sum(sbleu) / len(sbleu)
        chainer.report({self.key: sbleu})


def parse_args():
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_opts(parser)
    model_opts(parser)
    opts = parser.parse_args()
    return opts


def calculate_unknown_ratio(data):
    n_unk = sum((s == UNK_ID).sum() for s in data)
    total = sum(s.size for s in data)
    return n_unk / total


def encdec_convert(batch, device):
    xs, ys = zip(*batch)

    xs_block = convert.concat_examples(xs, device, padding=IGNORE)
    ys_block = convert.concat_examples(ys, device, padding=IGNORE)
    xp = cuda.get_array_module(xs_block)
    ref_block = xp.pad(ys_block, ((0, 0), (0, 1)),
                            'constant', constant_values=IGNORE)
    for num, seq in enumerate(ys):
        ref_block[num][len(seq)] = EOS_ID

    yis_block = xp.pad(ys_block, ((0, 0), (1, 0)),
                     'constant', constant_values=BOS_ID)
    return (xs_block, yis_block, ref_block)


def main():
    opts = vars(parse_args())
    os.mkdir(opts['save_dir'])
    save_path = os.path.join(opts['save_dir'], '{}.opts'.format(opts['model']))
    f = open(save_path, 'bw')
    pickle.dump(opts, f)

    # Load vocabulary and training data
    svocab = get_vocab(opts['train_src'], opts['src_vocabsize'], 
                       opts['src_minfreq'])
    tvocab = get_vocab(opts['train_tgt'], opts['tgt_vocabsize'], 
                       opts['tgt_minfreq'])
    train_src = word2id(opts['train_src'], svocab)
    train_tgt = word2id(opts['train_tgt'], tvocab)

    save_path = os.path.join(opts['save_dir'], 
                             '{}.svocab'.format(opts['model']))
    save_vocab(save_path, svocab)
    save_path = os.path.join(opts['save_dir'], 
                             '{}.tvocab'.format(opts['model']))
    save_vocab(save_path, tvocab)


    print('*** Details of training data ***')
    print('Source vocabulary size: %d' % len(svocab))
    print('Target vocabulary size: %d' % len(tvocab))
    print('Source data size: %d' % len(train_src))
    print('Target data size: %d' % len(train_tgt))
    print('Source data unknown ratio: %f' % calculate_unknown_ratio(train_src))
    print('Target data unknown ratio: %f' % calculate_unknown_ratio(train_tgt))
    print('')

    # Setup model
    print('Setup model')
    print('')
    model = Seq2seq(svocab, tvocab, opts)
    if opts['gpuid'] >= 0:
        cuda.get_device(opts['gpuid']).use()
        model.to_gpu(opts['gpuid'])

    # Setup optimizer
    if opts['optim'] == 'Adam':
        optimizer = chainer.optimizers.Adam()
    elif opts['optim'] == 'SGD':
        optimizer = chainer.optimizers.SGD()
    optimizer.setup(model)

    # Setup iterator
    train_data = [(s, t) for s, t in zip(train_src, train_tgt)
                   if opts['src_minlen'] <= len(s) <= opts['src_maxlen'] 
                   and opts['tgt_minlen'] <= len(t) <= opts['tgt_maxlen']]
    train_iter = chainer.iterators.SerialIterator(train_data, opts['batchsize'])

    # Setup updater
    updater = training.updaters.StandardUpdater(train_iter, optimizer, 
                converter=encdec_convert, device=opts['gpuid'])

    # Setup trainer
    trainer = training.Trainer(updater, (opts['epochs'], 'epoch'), 
                               opts['save_dir'])
    trainer.extend(extensions.LogReport(
        trigger=(opts['log_interval'], 'iteration')))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'elapsed_time']),
        trigger=(opts['log_interval'], 'iteration'))
    trainer.extend(extensions.PlotReport(
        ['main/loss'], x_key='iteration', file_name='loss.png'))

    trainer.extend(
        SaveModel(model, opts['save_dir'], opts['model']),
        trigger=(1, 'epoch'))

    if opts['valid_src'] and opts['valid_tgt']:
        valid_src = word2id(opts['valid_src'], svocab)
        valid_tgt = word2id(opts['valid_tgt'], tvocab)
        assert len(valid_src) == len(valid_tgt)

        print('*** Details of validation data ***')
        print('Source data size: %d' % len(valid_src))
        print('Target data size: %d' % len(valid_tgt))
        print('Source data unknown ratio: %f' \
              % calculate_unknown_ratio(valid_src))
        print('Target data unknown ratio: %f' \
              % calculate_unknown_ratio(valid_tgt))
        print('')

        valid_data = [(s, t) for s, t in zip(valid_src, valid_tgt)
                       if opts['src_minlen'] <= len(s) <= opts['src_maxlen'] 
                       and opts['tgt_minlen'] <= len(t) <= opts['tgt_maxlen']]

        if opts['report_sbleu']:
            trainer.extend(CalculateSBLEU(model, valid_data, 'valid/main/sbleu', 
                batch=opts['batchsize'], device=opts['gpuid'], 
                maxlen=opts['maxlen']), trigger=(1, 'epoch'))
            trainer.extend(extensions.PlotReport(['valid/main/sbleu'], 
               x_key='epoch', file_name='sbleu.png'))

    # Training
    print('start training')
    trainer.run()


if __name__ == '__main__':
    main()
