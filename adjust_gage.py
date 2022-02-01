"""Adjust USGS gage readings

This script will:
1. Convert from feet to meters
2. Apply general dataum correction
3. Apply gage datum correction

to USGS gage readings.
"""

import pandas as pd
import argparse
import pathlib
import itertools
import concurrent.futures

CORRECTIONS = None

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("corrections", type=pathlib.Path, help="Path to corrections CSV")
    parser.add_argument("data_dir", type=pathlib.Path, help="Path to USGS data directory")
    parser.add_argument("output_dir", type=pathlib.Path, help="Path to output directory")

    parser.add_argument("--overwrite-existing", dest="overwrite", 
                        action="store_true", help="Reprocess and overwrite files in output directory.")

    return parser.parse_args()


def ft_to_m(vals):
    return vals * 0.3048


def ft3_to_m3(vals):
    return vals * 0.3048**3


def load_corrections(corrections):
    """Setup worker environment.
    A global variable used here to avoid serializing and passing corrections
    file to each worker process"""
    global CORRECTIONS
    CORRECTIONS = pd.read_csv(corrections, index_col='Gauge ID').sort_index()


def adjust_station(path, out_path):
    data = pd.read_csv(path, low_memory=False)
    data = data.rename(columns={'gage height (ft)': 'gage height (m)',
                                'streamflow (ft^3/s)': 'streamflow (m^3/s)'})
    data['gage height (m)'] = ft_to_m(data['gage height (m)'])
    data['streamflow (m^3/s)'] = ft3_to_m3(data['streamflow (m^3/s)'])

    key = int(path.stem.split('.', 1)[0])
    if key in CORRECTIONS.index:
        adj = CORRECTIONS.loc[key, ['Datum correction (m)', 'Gage datum (m)']].sum()
        print("Adjusting", path)
        data['gage height (m)'] += adj
    
    print(" ->", out_path)
    try:
        data.to_csv(out_path, index=None)
    except KeyboardInterrupt:
        out_path.unlink()


def main(args):
    # Find all the compressed/uncompressed csv files in directory
    files = itertools.chain(args.data_dir.glob("*.csv"), args.data_dir.glob("*.csv.xz"))
    
    with concurrent.futures.ProcessPoolExecutor(initializer=load_corrections, initargs=(args.corrections,)) as executor:
        fs = []
        for f in files:
            out_path = args.output_dir/f.name
            if args.overwrite or not out_path.exists():
                fs.append(executor.submit(adjust_station, f, out_path))
        
        concurrent.futures.wait(fs, return_when=concurrent.futures.FIRST_EXCEPTION)

        

if __name__ == "__main__":
    args = get_options()
    main(args)