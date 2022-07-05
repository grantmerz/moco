import argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--nrangeup', default=1.0,type=float,
                    help='data normalization range')

parser.add_argument('--nrangelow', default=0.0, type=float,
                    help='data normalization range')

args=parser.parse_args()
nrange = [args.nrangelow,args.nrangeup]
print(nrange)
