# -*- coding: utf-8 -*-


def pretrain_opts(parser):
    group = parser.add_argument_group('Pre-raining')
    group.add_argument('--re-training', default=None,
        help='path to trained model')
    group.add_argument('--train', default='./samples/sample_train.txt',
        help='filename of the train data')
    group.add_argument('--valid', default=None,
        help='filename of the validation data')
    group.add_argument('--bptt-len', type=int, default=128,
        help='length of sequences for backpropagation through time')
    group.add_argument('--batch-size', type=int, default=64, 
        help='batch size')
    group.add_argument('--savedir', default='./checkpoints/pre-trained', 
        help='path to save models')
    group.add_argument('--max-epoch', type=int, default=30, 
        help='number of epochs')
    group.add_argument('--lr', type=float, default=0.25,
        help='learning rate')
    group.add_argument('--min-lr', type=float, default=1e-5, 
        help='minimum learning rate')
    group.add_argument('--clip', type=float, default=1.0,
        help='gradient cliping')
    group.add_argument('--gpu', action='store_true',
        help='whether gpu is used')
    group.add_argument('--optimizer', choices=['sgd', 'adam', 'adagrad'],
        default='sgd', help='optimizer')
    group.add_argument('--save-epoch', type=int, default=10)


def finetune_opts(parser):
    group = parser.add_argument_group('Training')
    group.add_argument('--finetune', default=None,
        help='path to pre-trained model')
    # group.add_argument('--re-training', default=None,
    #     help='path to trained model')
    group.add_argument('--train', default='./samples/sample_train.tsv',
        help='filename of the train data')
    group.add_argument('--valid', 
        default='./samples/sample_valid.tsv',
        help='filename of the validation data')
    group.add_argument('--src-minlen', type=int, default=0,
        help='minimum sentence length of source side')
    group.add_argument('--tgt-minlen', type=int, default=0,
        help='minimum sentence length of target side')
    group.add_argument('--src-maxlen', type=int, default=100,
        help='maximum sentence length of source side')
    group.add_argument('--tgt-maxlen', type=int, default=100,
        help='maximum sentence length of target side')
    group.add_argument('--batch-size', type=int, default=64, 
        help='batch size')
    group.add_argument('--savedir', default='./checkpoints/fine-tuned', 
        help='path to save models')
    group.add_argument('--max-epoch', type=int, default=30, 
        help='number of epochs')
    group.add_argument('--lr', type=float, default=0.25,
        help='learning rate')
    group.add_argument('--min-lr', type=float, default=1e-5, 
        help='minimum learning rate')
    group.add_argument('--clip', type=float, default=1.0,
        help='gradient cliping')
    group.add_argument('--gpu', action='store_true',
        help='whether gpu is used')
    group.add_argument('--optimizer', choices=['sgd', 'adam', 'adagrad'],
        default='sgd', help='optimizer')
    group.add_argument('--save-epoch', type=int, default=10)


def model_opts(parser):
    group = parser.add_argument_group('Model\'s hyper-parameters')
    group.add_argument('--embed-dim', type=int, default=512,
        help='dimension of word embeddings of decoder')
    group.add_argument('--embed-path', default=None,
        help='pre-trained word embeddings')
    group.add_argument('--min-freq', type=int, default=0,
        help='map words appearing less than threshold times to unknown')
    group.add_argument('--hidden-dim', type=int, default=2048,
        help='number of hidden units per decoder layer')
    group.add_argument('--layers', type=int, default=6,
        help='number of layers')
    group.add_argument('--heads', type=int, default=16,
        help='number of attention heads')
    group.add_argument('--dropout', type=float, default=0.2,
        help='dropout applied to layers (0 means no dropout)')
    group.add_argument('--activation-dropout', type=float, default=0.1,
        help='dropout after activation fucntion in self attention')
    group.add_argument('--attention-dropout', type=float, default=0.1,
        help='dropout in self attention')
    group.add_argument('--normalize-before', action='store_true',
        help='apply layernorm before each decoder block')
    return group


def generate_opts(parser):
    group = parser.add_argument_group('Generation')
    group.add_argument('--model', 
        default='./checkpoints/checkpoint_best.pt',
        help='model file for translation')
    group.add_argument('--input', default='./samples/sample_test.txt',
        help='input file')
    group.add_argument('--batch-size', type=int, default=32,
        help='batch size')
    group.add_argument('--maxlen', type=int, default=100,
        help='maximum length of output sentence')
    group.add_argument('--gpu', action='store_true',
        help='whether gpu is used')
    return group
