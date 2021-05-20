import argparse

parser = argparse.ArgumentParser(description='...')
parser.add_argument('-l','--layer', type=int, nargs='+', required=True, action='append', help='layer list')
parser.add_argument("--activation", nargs="+", default= ["relu"])

args = parser.parse_args()

print(args.layer)
print(args.activation)