"""
Author: Ryan Grout

Adjust USGS gage readings

This script will:
1. Convert from feet to meters
2. Apply general dataum correction
3. Apply gage datum correction

to USGS gage readings.
"""

import pandas as pd
import numpy as np
import argparse
import pathlib
import itertools
import concurrent.futures

CORRECTIONS = None

COL_MAP = {'streamflow (ft^3/s)': 'streamflow (m^3/s)',
            'gage height (ft)': 'gage height (m)',
            "Elevation ocean/est (ft NAVD88)": "Elevation ocean/est (m NAVD88)",
            "Water Level (ft)": "Water Level (m)"}
            

def real_stem(path):
    return path.name[:len(path.name) - sum(map(len, path.suffixes))]
    

def with_stem(path, stem):
    return path.with_name(stem + ''.join(path.suffixes))
    

def adjusted_fn(path):
    key = path.name.split('.', 1)[0]
    if key in CORRECTIONS.index:
        adj = CORRECTIONS.loc[key, ['Datum correction (m)', 'Gage datum (m)']].sum()
        if adj != 0:
            return with_stem(path, f"{real_stem(path)}_NAVD88")
    return path
    

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("corrections", type=pathlib.Path, help="Path to corrections CSV")
    parser.add_argument("data_dir", type=pathlib.Path, help="Path to USGS data directory")
    parser.add_argument("output_dir", type=pathlib.Path, help="Path to output directory")

    parser.add_argument("--overwrite-existing", dest="overwrite", 
                        action="store_true", help="Reprocess and overwrite files in output directory.")
    parser.add_argument("--only-adjusted", dest="write_adjusted",
                        action="store_true", help="Only write stations that have a correction applied.")
    parser.add_argument("--only-summary", dest="summary",
                        action="store_true", help="Only compute stats on adjusted stations.")

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
    CORRECTIONS = pd.read_csv(corrections, index_col='Gage ID',
                            usecols=['Gage ID', 'Datum correction (m)', 'Gage datum (m)']).sort_index()
    # Check to make sure index is unique
    if CORRECTIONS.index.has_duplicates:
        print(CORRECTIONS.index[CORRECTIONS.index.duplicated()].tolist())
        raise RuntimeError("Corrections has duplicate gage ids")


def adjust_station(path, out_path, only_write_adjusted=False):
    key = path.name.split('.', 1)[0]
    if key in CORRECTIONS.index:
        print("Adjusting", path)
        adj = CORRECTIONS.loc[key, ['Datum correction (m)', 'Gage datum (m)']].sum()
    elif only_write_adjusted:
        return
    else:
        adj = 0

    data = pd.read_csv(path, low_memory=False)
    data = data.rename(columns=COL_MAP)

    for c in COL_MAP.values():
        if c in data:
            data[c] = ft_to_m(data[c])
    print(" "*80, end="\r")

    if 'gage height (m)' in data.columns:
        data['gage height (m)'] += adj
        
    print(" ->", out_path)
    try:
        data.to_csv(out_path, index=None)
    except (KeyboardInterrupt, SystemExit):
        out_path.unlink()
    return (key, compute_stats(data))
    
    
def station_stats(path):
    """Compute stats on an already adjusted station"""
    key = path.name.split('.', 1)[0]
    data = pd.read_csv(path, low_memory=False)
    return (key, compute_stats(data))
    

def compute_stats(df):
    """Compute min, max, and median of corrected columns in df"""
    return df[df.columns.intersection(COL_MAP.values())].describe()


def main(args):    
    load_corrections(args.corrections)

    # create output_directory if it doesn't exist
    args.output_dir.mkdir(parents=True)
    
    #with concurrent.futures.ThreadPoolExecutor(initializer=load_corrections, initargs=(args.corrections,)) as executor:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        fs = []
        
        if args.summary:
            files = list(args.output_dir.glob("*.csv.xz"))
            print("Computing summary statistics on existing output...")
            for f in files:
                fs.append(executor.submit(station_stats, f))
        else:
            files = list(args.data_dir.glob("*.csv.xz"))
            print("Converting/adjusting measurements...")
            for f in files:
                out_path = adjusted_fn(args.output_dir/f.name)
                if args.overwrite or not out_path.exists():
                    fs.append(executor.submit(adjust_station, f, out_path, only_write_adjusted=args.write_adjusted))
                    #adjust_station(f, out_path, only_write_adjusted=args.write_adjusted)
            
        ntasks = len(fs)
        #summary = []
        with open(args.output_dir/"summary_stats.txt", "w") as out:                
            for i, f in enumerate(concurrent.futures.as_completed(fs)):
                rv = f.result()
                print("finished task", i, "of", ntasks, f"{round(100*(i/ntasks))}%")
                if rv:
                    key, stats = rv
                    print("Writing", key)
                    out.write(f"=== Station {key}:")
                    out.write("\n")
                    stats.to_string(buf=out, na_rep='--')
                    out.write("\n\n")
                    out.flush()
                    
            
    #print("Processing summary stats...")
    #summary = list(zip(*summary))
    #PP = pd.concat([pd.concat(s, axis=1) for s in summary[1:]], keys=['min', 'max', 'mean'])
    #PP.columns = pd.Index(summary[0])
    #out_summary = args.output_dir/"summary_stats.csv"
    #PP.T.to_csv(out_summary)
    #print("Summary written to", out_summary) 
    
        

if __name__ == "__main__":
    args = get_options()
    main(args)