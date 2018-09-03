import numpy
import chainer.links as L
import chainer.functions as F
import chainer
from chainer import cuda
from Const import BOS_ID, EOS_ID, IGNORE


def get_argnbest(vecs, n, reverse=False):
    if type(vecs) == chainer.variable.Variable:
        vecs = vecs.data
    xp = cuda.get_array_module(vecs)
    ids = xp.argsort(vecs, axis=1)
    if reverse:
        tops = [id[::-1][:n].tolist() for id in ids]
    else:
        tops = [id[:n].tolist() for id in ids]
    return tops


def sequence_linear(W, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = W(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class GlobalAttention(chainer.Chain):
    def __init__(self, units, score):
        super(GlobalAttention, self).__init__()
        with self.init_scope():
            self.units = units
            self.score = score

            if score == 'general':
                self.wg = L.Linear(units, units)
            elif score == 'concat':
                self.wa = L.Linear(2*units, 1)

    def __call__(self, oys, oxs):
        self.bs, self.xlen, _ = oxs.shape
        _, self.ylen, _ = oys.shape

        # select scoring method
        if self.score == 'dot':
            scores = self.dot(oys, oxs)
        elif self.score == 'general':
            scores = self.general(oys, oxs)

        # calculate context vectors
        oxs = F.broadcast_to(oxs, (self.ylen, self.bs, self.xlen, self.units))
        oxs = F.transpose(oxs, (1, 0, 2, 3))
        scores = F.broadcast_to(scores, 
                    (self.units, self.bs, self.ylen, self.xlen))
        scores = F.transpose(scores, axes=(1, 2, 3, 0))
        ct = F.sum(oxs*scores, axis=2)
        return ct

    def dot(self, oys, oxs):
        oxs = F.broadcast_to(oxs, (self.ylen, self.bs, self.xlen, self.units))
        oys = F.broadcast_to(oys, (self.xlen, self.bs, self.ylen, self.units))
        oxs = F.transpose(oxs, (1, 0, 2, 3))
        oys = F.transpose(oys, (1, 2, 0, 3))
        scores = F.sum(oxs*oys, axis=3)
        scores = F.softmax(scores, axis=2)
        return scores

    def general(self, oys, oxs):
        oxs = F.stack(sequence_linear(self.wg, oxs))
        scores = self.dot(oys, oxs)
        return scores


class Encoder(chainer.Chain):
    def __init__(self, vocab, rnn, layers, units, dout):
        super(Encoder, self).__init__()
        with self.init_scope():
            vocabsize = len(vocab) + 3 # 3 means number of special tags
            initW = None
            self.emb = L.EmbedID(vocabsize, units, initW, IGNORE)
            self.rnn = L.NStepLSTM(layers, units, units, dout)

    def nstep(self, xs, reverse):
        if reverse:
            xs = [x[::-1] for x in xs]
        emb_xs = sequence_linear(self.emb, xs)
        hx, cx, oxs = self.rnn(None, None, emb_xs)
        return hx, cx, oxs


class Decoder(chainer.Chain):
    def __init__(self, vocab, rnn, layers, units, dout):
        super(Decoder, self).__init__()
        with self.init_scope():
            vocabsize = len(vocab) + 3 # 3 means number of special tags
            initW = None
            self.emb = L.EmbedID(vocabsize, units, initW, IGNORE)
            self.rnn = L.NStepLSTM(layers, units, units, dout)
            self.wo = L.Linear(units, vocabsize)

    def nstep(self, ys, hx, cx):
        # Decode
        emb_ys = sequence_linear(self.emb, ys)
        _, _, oys = self.rnn(hx, cx, emb_ys)
        oys = sequence_linear(self.wo, oys)
        return oys

    def onestep(self, ys, hx, cx):
        bs = len(ys)
        emb_ys = self.emb(ys)
        emb_ys = F.split_axis(emb_ys, bs, 0)
        hy, cy, oys = self.rnn(hx, cx, emb_ys)
        oys = self.wo(F.concat(oys, axis=0))
        return hy, cy, oys


class DecoderAttn(chainer.Chain):
    def __init__(self, vocab, rnn, layers, units, dout, attn, score, feed):
        super(DecoderAttn, self).__init__()
        with self.init_scope():
            vocabsize = len(vocab) + 3 # 3 means number of special tags
            initW = None
            self.emb = L.EmbedID(vocabsize, units, initW, IGNORE)
            input_size = 2 * units if feed else units
            self.rnn = L.NStepLSTM(layers, input_size, units, dout) 
            self.wo = L.Linear(units, vocabsize)

            # Setup attentional mechanism
            self.feeding = feed
            self.attn = GlobalAttention(units, score)
            self.wc = L.Linear(2 * units, units)

    def nstep(self, ys, hx, cx, oxs):
        # Decode
        emb_ys = sequence_linear(self.emb, ys)
        _, _, oys = self.rnn(hx, cx, emb_ys)

        # calculate context vectors
        oxs = F.stack(oxs)
        oys = F.stack(oys)
        cts = self.attn(oys, oxs)
        cs = F.concat((oys, cts), axis=2)
        hts = F.tanh(F.stack(sequence_linear(self.wc, cs)))
        oys = sequence_linear(self.wo, hts)
        return oys

    def onestep(self, ys, hx, cx, oxs, hts):
        bs = len(ys)
        emb_ys = self.emb(ys)

        if self.feeding:
            hts = F.stack(hts)
            emb_ys = F.expand_dims(emb_ys, axis=1)
            emb_ys = F.concat((emb_ys, hts), axis=2)
            hy, cy, oys = self.rnn(hx, cx, F.separate(emb_ys))
        else:
            emb_ys = F.split_axis(emb_ys, bs, 0)
            hy, cy, oys = self.rnn(hx, cx, emb_ys)

        oys = F.stack(oys)
        oxs = F.stack(oxs)
        cts = self.attn(oys, oxs)
        cs = F.concat((oys, cts), axis=2)
        hts = F.tanh(F.stack(sequence_linear(self.wc, cs)))
        oys = self.wo(F.concat(hts, axis=0))
        return hy, cy, oys, hts


class Seq2seq(chainer.Chain):
    def __init__(self, svocab, tvocab, opts):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.encoder = Encoder(svocab, opts['rnn'], opts['layers'], 
                                opts['units'], opts['dout'])

            if opts['attn'] == 'disuse':
                self.decoder = Decoder(tvocab, opts['rnn'], opts['layers'], 
                                       opts['units'], opts['dout'])
            else:
                self.decoder = DecoderAttn(tvocab, opts['rnn'], opts['layers'], 
                                    opts['units'], opts['dout'], opts['attn'], 
                                    opts['score'], opts['input_feeding'])

        self.reverse = opts['reverse_encoding']
        self.feeding = opts['input_feeding']
        self.use_attn = False if opts['attn'] == 'disuse' else True
        self.units = opts['units']

    def __call__(self, xs, yis, ref):
        # Padding
        if self.feeding:
            loss = self.feeding_loss(xs, yis, ref)
        else:
            loss = self.loss(xs, yis, ref)
        chainer.report({'loss':loss.data}, self)
        return loss

    def loss(self, xs, yis, ref):
        bs = len(xs)
        hx, cx, oxs = self.encoder.nstep(xs, reverse=self.reverse)

        if self.use_attn:
            oys = self.decoder.nstep(yis, hx, cx, oxs)
        else:
            oys = self.decoder.nstep(yis, hx, cx)
        ref = F.concat(ref, axis=0)
        oys = F.concat(oys, axis=0)
        return F.sum(F.softmax_cross_entropy(oys, ref, reduce='no')) / bs

    def feeding_loss(self, xs, yis, ref):
        bs = len(xs)
        ylen = yis.shape[1]
        hx, cx, oxs = self.encoder.nstep(xs, reverse=self.reverse)
        hts = [self.xp.zeros(self.units, 'f').reshape(1, self.units)
               for _ in range(bs)]
        loss = chainer.Variable(self.xp.zeros((), 'f'))
        for i in range(ylen):
            hx, cx, oys, hts = self.decoder.onestep(yis[:, i], hx, cx, oxs, hts)
            loss += F.sum(F.softmax_cross_entropy(oys, ref[:,i], reduce='no'))
        return loss / bs

    def translate(self, xs, maxlen):
        bs = len(xs)
        xs = chainer.dataset.concat_examples(xs, padding=IGNORE)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            h, c, oxs = self.encoder.nstep(xs, reverse=self.reverse)
            result = []
            ws = self.xp.full(bs, BOS_ID, numpy.int32)
            
            ht = [self.xp.zeros(self.units, 'f').reshape(1, self.units)
                  for _ in range(bs)] if self.feeding else None
 
            for _ in range(maxlen):
                if self.use_attn:
                    h, c, o, ht = self.decoder.onestep(ws, h, c, oxs, ht)
                else:
                    h, c, o = self.decoder.onestep(ws, h, c)
                ws = self.xp.argmax(o.data, axis=1).astype(numpy.int32)
                result.append(ws)

        result = cuda.to_cpu(self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

        # Remove EOS tags
        outputs = []
        for seq in result:
            inds = numpy.argwhere(seq == EOS_ID)
            if len(inds) > 0:
                seq = seq[:inds[0, 0]]
            outputs.append(seq.tolist())
        return outputs


    def beam(self, xs, ys, maxlen, beamsize, n_cands, ranking):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            h, c, oxs = self.encoder.nstep(xs, reverse=self.reverse)

            # Initiarization
            ht = [self.xp.zeros(self.units, 'f').reshape(1, self.units)] \
                  if self.feeding else None

            que = [(0.0, [BOS_ID], h, c, ht)]

            # Beam search
            for _ in range(maxlen):
                if all(map(lambda s: s[1][-1] == EOS_ID, que)):
                    break
                new_que = [] 
                for score, seq, h, c, ht in que:
                    if seq[-1] == EOS_ID:
                        new_que.append((score, seq, h, c, ht))
                    else:
                        # decode
                        w = self.xp.array([seq[-1]], self.xp.int32)
                        if self.use_attn:
                            h, c, o, ht = self.decoder.onestep(w, h, c, oxs, ht)
                        else:
                            h, c, o = self.decoder.onestep(w, h, c)
                        o = -F.log_softmax(o)
                        nbest_ids = get_argnbest(o, beamsize)[0]
                        
                        # calclate log likelihood
                        for index in nbest_ids:
                            new_score = score + float(o[0][index].data)
                            new_seq = seq + [index]
                            new_que.append((new_score, new_seq, h, c, ht))

                # sort in the new_queue of the higher likelihood
                new_que.sort(key=lambda x: x[0]/(len(x[1]) - 1))
                que = new_que[:beamsize]

        # Remove EOS and BOS tags
        hyps = [que[i][1][1:-1] if que[i][1][-1] == EOS_ID 
                   else que[i][1][1:] for i in range(beamsize)]

        # ranking
        if ranking == 'sbleu':
            hyps = self.sbleu_ranking(hyps, ys)
        return hyps[:n_cands]

