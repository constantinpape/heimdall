#!/usr/bin/env python

import argparse
from .. import view_container


def tobool(inp):
    return inp.lower() in ('y', '1', 'ok', 't')


parser = argparse.ArgumentParser(description='Display datasets in h5 or n5/zarr container.')
parser.add_argument('path', type=str, help='path to container')
parser.add_argument('--ndim', type=int, default=3,
                    help='expected number of dimensions')
parser.add_argument('--exclude_names', type=str, nargs='+', default=None,
                    help='names of datasets that will not be loaded')
parser.add_argument('--include_names', type=str, nargs='+', default=None,
                    help='names of datasets that will ONLY be loaded')
parser.add_argument('--load_into_memory', type=tobool, default='n',
                    help='whether to load all data into memory')
parser.add_argument('--n_threads', type=int, default=1,
                    help='number of threads used by z5py')


def main():
    args = parser.parse_args()
    view_container(args.path, args.ndim,
                   args.exclude_names, args.include_names,
                   args.load_into_memory, args.n_threads)


if __name__ == '__main__':
    main()
