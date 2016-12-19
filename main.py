import numpy as np
import chainer
from chainer import optimizers
import chainer.links as L
import chainer.functions as F
import dnc
import argparse

class MyModel(chainer.Chain):
    def __init__(self, n_vocab, embed_dim, n_locations, memory_width,
            n_read_heads, n_units_of_hidden_layer, n_layers, norecurrent=False):
        super(MyModel, self).__init__(
                emb = L.EmbedID(n_vocab, embed_dim),
                dnc = dnc.DefaultDNC(n_locations, memory_width,
                    n_read_heads, embed_dim, embed_dim,
                    n_units_of_hidden_layer, n_layers, norecurrent),
                emb_inv = L.Linear(embed_dim, n_vocab)
                )

    def __call__(self, x):
        e = self.emb(x)
        y = self.dnc(e)
        z = self.emb_inv(y)
        return z

    def reset_state(self):
        self.dnc.reset_state()

def compute_loss(model, sample_seq):
    batch_size = 1
    loss = 0
    seq_len = len(sample_seq)
    for i in range(seq_len):
        x, t = sample_seq[i]
        x0 = chainer.Variable(np.array([x]*batch_size, dtype=np.int32))
        y0 = model(x0)
        if not i < (seq_len / 2):
            loss += F.softmax_cross_entropy(y0, np.array([t]*batch_size, dtype=np.int32))
    return loss

def draw_sample(n_vocab, variable=True, max_length=7):
    if variable:
        length = np.random.randint(1, max_length+1)
    else:
        length = max_length
    a = 1 + np.random.randint(n_vocab-1, size=length)
    b = zip(a, a)
    c = zip([0]*length, a)
    d = b + c
    return d

def test(model, n_vocab, max_length):
    sample_seq = draw_sample(n_vocab, variable=False, max_length=max_length)
    rs = []
    model.reset_state()
    for i in range(len(sample_seq)):
        x = sample_seq[i][0]
        y = model(np.array([x], dtype=np.int32))
        r = F.softmax(y).data.argmax(axis=1)[0]
        rs.append((x, r))
    return rs

def main():
    parser = argparse.ArgumentParser(description="An experiment of Differentiable Neural Computer : Echo tasks of variable length sequences")

    parser.add_argument('--seq_len', type=int, default=5, metavar='N',
            help='length of a sequence used by learning')
    parser.add_argument('--n_vocab', type=int, default=11, metavar='N',
            help='number of vocabulary')
    parser.add_argument('--embed_dim', type=int, default=4, metavar='N',
            help="dimension of vacabulary's inner representation")
    parser.add_argument('--n_locations', type=int, default=16, metavar='N',
            help='number of memory slots')
    parser.add_argument('--memory_width', type=int, default=32, metavar='N',
            help='number of elements in a memory slot')
    parser.add_argument('--n_read_heads', type=int, default=2, metavar='N',
            help='number of read heads')
    parser.add_argument('--n_layers', type=int, default=3, metavar='N',
            help='number of hidden layers of the DNC controller')
    parser.add_argument('--n_units', type=int, default=50, metavar='N',
            help="number of units the DNC controller's hidden layer uses")
    parser.add_argument('--n_iter', type=int, default=100000, metavar='N',
            help='number of learning iteration')
    parser.add_argument('--gpu', type=int, default=-1, metavar='Z',
            help='GPU id')
    parser.add_argument('--norecurrent', default=False, action='store_true',
            help="disable memory ability of the DNC controller (memorize by only it's external memory)")

    args = parser.parse_args()
    print("args:{}".format(vars(args)))

    model = MyModel(args.n_vocab, args.embed_dim, args.n_locations, args.memory_width,
            args.n_read_heads, args.n_units, args.n_layers, norecurrent=args.norecurrent)
    #optimizer = optimizers.RMSprop() # The DNC paper uses RMSprop optimizer. But it seems like it doesn't converge on our experiments.
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    loss_acc = 0
    for i in range(args.n_iter):
        if (i+1) % 100 == 0 or i+1 == args.n_iter:
            print("iter:{}".format(i+1))
            print("loss:{}".format(loss_acc))
            print("result:{}".format(test(model, args.n_vocab, args.seq_len)))
            loss_acc = 0

        sample_seq = draw_sample(args.n_vocab, max_length=args.seq_len)
        optimizer.target.reset_state()
        loss = compute_loss(optimizer.target, sample_seq)
        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()

        loss_acc += loss.data

main()
