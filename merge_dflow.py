"""
Concatenate dflow files into a single file, but in a memory efficient way.
"""

import xarray as xr
import argparse
import pathlib


def main(args):
    datasets = []
    for f in args.files:
        datasets.append(xr.open_dataset(f))

    merged = xr.concat(datasets, 'time', data_vars='minimal', coords='minimal')
    dups = merged.time.to_index().duplicated()
    merged = merged.sel({'time': ~dups})
    merged.to_netcdf(args.output)

    for f in datasets:
        f.close()


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("files", nargs='+', type=pathlib.Path, help='Files to merge (in order of precedence)')
    parser.add_argument('-o', '--output', type=pathlib.Path, help='Output')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_options()
    main(args)