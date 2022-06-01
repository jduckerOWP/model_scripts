"""
Author: Ryan Grout
"""

import pandas as pd
import numpy as np
import requests
import argparse
from itertools import groupby, zip_longest
import sys
from functools import partial
import pathlib
import random
import pdb

import asyncio
import concurrent.futures

BASE_URL = "http://waterservices.usgs.gov/nwis/iv"
HEADERS = {"Accept-Encoding": "gzip, compress"}
PARAMS = {"format": "json", "startDT": "2003-01-01" , "parameterCd": [], "siteStatus": "all"}

#COLUMN_ORDER = ['siteCode', 'streamflow (ft^3/s)', 'sf qualifiers', 'gage height (ft)', 'gh qualifiers', 'longitude', 'latitude']
#COLUMN_ORDER = ['siteCode', 'Precip Total (in)', 'pt qualifiers', 'Physical Precip Total (in/wk)', 'ppr qualifiers', 'longitude', 'latitude']
#COLUMN_ORDER = ['siteCode', 'Elevation ocean/est (ft NAVD88)', 'elv qualifiers', 'longitude', 'latitude']

VAR_CODES = {'00060': 'streamflow (ft^3/s)', '00065': 'gage height (ft)',
             '00045': 'Precip Total (in)', 
             '00046': 'Physical Precip Total (in/wk)',
             '62620': "Elevation ocean/est (ft NAVD88)",
             '62615': "Water Level (ft)"}
ABBREV = {'00060': 'sf', '00065': 'gh',
          '00045': 'pt', '00046': 'ppr', '62620': 'elv', '62615': 'wl'}
GROUP_SIZE = 1


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    chunks = zip_longest(*args, fillvalue=fillvalue)
    for chunk in chunks:
        yield filter(None, chunk)

def join_params(params, sep=','):
    joined_params = {}
    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            joined_params[k] = ','.join(v)
        else:
            joined_params[k] = v
    return joined_params


def make_df(variable):
    # Columns 
    # dateTime (UTC), streamflow (ft^3/s), sf qualifiers, gage height (ft), gh qualifiers, siteCode, longitude, latitude
    
    # Create the dataframe and cast to proper types.
    df = pd.DataFrame.from_records(variable['values'][0]['value'])
    #df['dateTime'] = pd.to_datetime(df['dateTime'], utc=True, format="%Y-%m-%dT%H:%M:%S.%f%z")
    df['dateTime'] = pd.to_datetime(df['dateTime'], utc=True, infer_datetime_format=True)
    duptimes = df['dateTime'].duplicated()
    if not duptimes.empty:
        df = df.loc[~duptimes]
    
    df = df.set_index('dateTime', verify_integrity=True)
    df.index = df.index.set_names("Date (utc)")
    df['value'] = pd.to_numeric(df['value'])
    df['qualifiers'] = df['qualifiers'].astype('string')

    # set nodata values to nan
    nodata = variable['variable']['noDataValue']
    df[df['value'] == nodata] = np.nan
    
    vcd = variable['variable']['variableCode'][0]['value']
    renames = {'value': VAR_CODES[vcd], 'qualifiers': f"{ABBREV[vcd]} qualifiers"}
    df = df.rename(columns=renames)
    
    coords = variable['sourceInfo']['geoLocation']['geogLocation']
    keys = {'siteCode': variable['sourceInfo']['siteCode'][0]['value'],
            'longitude': float(coords['longitude']),
            'latitude': float(coords['latitude'])}
    return keys, df

def get_values(resp):
    dfs = []
    for v in resp['value']['timeSeries']:
        if v and v['values'][0]['value']:
            dfs.append(make_df(v))
    return dfs

def groupby_sitecode_join(dfs):
    keyfunc = lambda x: x[0]['siteCode']
    dfs = sorted(dfs, key=keyfunc)
    
    _dfs = []
    for code, g in groupby(dfs, key=keyfunc):
        frames = tuple(g)
        try:
            df = pd.concat((x[1] for x in frames), axis='columns', copy=False)
        except:
            breakpoint()
        lon = frames[0][0]['longitude']
        lat = frames[0][0]['latitude']
        df[['longitude', 'latitude', 'siteCode']] = lon, lat, code
        _dfs.append(df)
    return pd.concat(_dfs, axis='index', copy=False)


def process_request(result, output):
    if result.ok:
        #print("Processing a result")
        dfs = get_values(result.json())
        if dfs:
            DF = groupby_sitecode_join(dfs)
            print("Processing data...", len(DF))
            #expected_cols = set()
            #for p in PARAMS['parameterCd']:
            #    expected_cols.update((VAR_CODES[p], f"{ABBREV[p]} qualifiers"))
            #missing_cols = list(expected_cols.difference(DF.columns))
            #if missing_cols:
            #    DF[missing_cols] = pd.NA
            DF = DF.sort_index()
            for code, df in DF.groupby('siteCode'):
                fname = output / f"US{code}.csv.xz"
                print("Writing to", fname)
                try:
                    df.to_csv(fname)
                except KeyboardInterrupt:
                    print("Removing partial file", fname.absolute())
                    fname.unlink()
                    break
            else:
                print("Finished processing", len(DF), "data points")

def make_request2(params):
    print("Requesting", params['sites'].count(',') + 1, "sites")
    return requests.get(BASE_URL, params=params, headers=HEADERS)

def request_task(params, output):
    response = make_request2(params)
    process_request(response, output)


async def download_data(sites, output):
    tasks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for s in grouper(sites, GROUP_SIZE):
            sitecodes = list(s)
            params = PARAMS.copy()
            params['sites'] = sitecodes
            tasks.append(pool.submit(request_task, join_params(params), output))
            #request_task(join_params(params), output)
        
        for t in concurrent.futures.as_completed(tasks):
            if exc := t.exception():
                raise exc
        #waited_tasks = concurrent.futures.wait(tasks, return_when=concurrent.futures.FIRST_EXCEPTION)
        
        print("Done awaiting futures")
    print("Shut down pool")


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('sites', type=pathlib.Path, help='List of sites to download')
    parser.add_argument('-p', '--param', dest='params', action='append', help="Parameters to request")
    parser.add_argument('-o', '--output', type=pathlib.Path, default=pathlib.Path(), help="output directory")

    args = parser.parse_args()
    return args
        
if __name__ == "__main__":
    args = get_options()
    PARAMS['parameterCd'] = args.params

    sites = []

    # Check if args.output exists. Create if not
    if not args.output.exists():
        args.output.mkdir(parents=True)
        
    print("Writing to", args.output)
    print(PARAMS)
    existing = set(x.name for x in args.output.glob('*.csv.xz'))
    with open(args.sites, 'r') as fi:
        for L in map(str.strip, fi):
            if L.startswith('#'):
                continue
            if L.isnumeric() and f"{L}.csv.xz" not in existing:
                sites.append(L)
    print("Read", len(sites), "sites from", args.sites)
    random.shuffle(sites)
    asyncio.run(download_data(sites, args.output))
    

