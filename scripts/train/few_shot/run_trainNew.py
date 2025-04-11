import argparse
import os
from trainNew import main

parser = argparse.ArgumentParser(description='Train Prototypical Network on XBD')

# Minimal set of essential arguments
parser.add_argument('--data.root', type=str, default='../../../data', help='Root folder containing train/val/test folders')
parser.add_argument('--model.x_dim', type=str, default='6,128,128', help='Input dimensions (default: 6,128,128)')
parser.add_argument('--model.hid_dim', type=int, default=64, help='Hidden layer size')
parser.add_argument('--model.z_dim', type=int, default=64, help='Latent dimension')
parser.add_argument('--log.exp_dir', type=str, default='results', help='Directory to save logs and models')
parser.add_argument('--log.fields', type=str, default='loss,acc', help='Fields to log')

# Few-shot episode settings
parser.add_argument('--data.way', type=int, default=2, help='Number of classes per episode')
parser.add_argument('--data.shot', type=int, default=5, help='Number of support examples per class')
parser.add_argument('--data.query', type=int, default=5, help='Number of query examples per class')

# Training config
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
parser.add_argument('--decay_every', type=int, default=20, help='StepLR decay every N epochs')
parser.add_argument('--optim_method', type=str, default='Adam', help='Optimizer method')
parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')

args = vars(parser.parse_args())

# Resolve data.root to an absolute path
# Leave as-is (no conversion to absolute path)
pass
main(args)