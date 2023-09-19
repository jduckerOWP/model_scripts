
import pathlib
import argparse
import json
import numpy as np
import netCDF4 as nc
import cftime
import pandas as pd
import datetime
from itertools import groupby, chain
from collections import defaultdict


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=pathlib.Path, help="Output directory")
    parser.add_argument("nwm_data", type=pathlib.Path, help="NWM CHRTOUT data")
    parser.add_argument("sources", type=pathlib.Path, help="SCHISM sources json")
    parser.add_argument("sinks", type=pathlib.Path, help="SCHISM sinks json")
    #parser.add_argument('source_sink', type=pathlib.Path, help="Schism Source/Sink list")

    args = parser.parse_args()
    return args

def fn_hourly(key):
    dt = datetime.datetime.strptime(key.name.split('.', 1)[0], "%Y%m%d%H%M")
    return dt

def groupby_hour(seq):
    rv = defaultdict(list)
    for k, i in groupby(seq, key=fn_hourly):
        rv[k].extend(i)

    #sort keys
    rv_sorted = {}
    for k in sorted(rv.keys()):
        rv_sorted[k] = rv[k]
    return rv_sorted

def binary_isin(elements, test_elements, assume_sorted=False, return_indices=False):
    """Test if values of elements are present in test_elements.
    Returns a boolean array: True if element is present in test_elements, False otherwise.
    If return_indices=True, return an array of indexes that transforms the elements of
    test_elements to same order of elements (for the elements that are present in test_elements).
    ie, for every True in the returned mask, also return the index such that test_elements[indices] == elements[mask]
    The method is usually slower than using np.isin for unsorted test_elements.
    However, the returns of this method can still be useful.
    """
    elements = np.asarray(elements)
    test_elements = np.asarray(test_elements)

    if assume_sorted:
        idxs = np.searchsorted(test_elements, elements)
    else:
        asorted = test_elements.argsort()
        idxs = np.searchsorted(test_elements, elements, sorter=asorted)

    valid = idxs != len(test_elements)
    test_selector = idxs[valid]
    if not assume_sorted:
        test_selector = asorted[test_selector]

    mask = np.zeros(elements.shape, dtype=bool)
    breakpoint()
    mask[valid] = test_elements[test_selector] == elements[valid]

    #indices are the array of indexes that transform the elements in
    # test_elements to match the order of elements
    if return_indices:
        return mask, test_selector[mask[valid]]
    else:
        return mask

def extract_offsets(current_netcdf_filename, selected_ids):
    """Return a mask on feature_id for the selected_ids
    Args:
        current_netcdf_filename (str): NetCDF file to read
        selected_ids (list_like): ids to select.
    Returns:
        (ndarray): Indexes *in order* of selected comm_ids.
    """
    with nc.Dataset(current_netcdf_filename) as ncdata:
        feature_id_index = np.ma.getdata(ncdata['feature_id'][:])
        print("Size of feature id data is ", len(feature_id_index))
        fmask, fidx = binary_isin(feature_id_index, selected_ids, return_indices=True)
        fvals = feature_id_index[fmask]
        return fmask, fvals, fidx


def expand_keys(d):
    for k, v in d.items():
        for i in range(len(v)):
            yield k

def main(args):
    # read in list of source and sink reaches
    with open(args.sources) as f:
        sources = json.load(f)

    with open(args.sinks) as f:
        sinks = json.load(f)

    nwm_sources = list(map(int, chain.from_iterable(sources.values())))
    nwm_sinks = list(map(int, chain.from_iterable(sinks.values())))

    # build vsource and vsink arrays
    chrtout_files = sorted(args.nwm_data.glob('*CHRTOUT*'))
    chrtout_groups = groupby_hour(chrtout_files)

    # if any(len(v) > 1 for v in chrtout_groups.values()):
    #     vsource = np.zeros((len(chrtout_groups)+1, len(soids)))
    #     vsink = np.zeros((len(chrtout_groups)+1, len(siids)))
    # else:
    #     vsource = np.zeros((len(chrtout_groups), len(soids)))
    #     vsink = np.zeros((len(chrtout_groups), len(siids)))

    # USE FIRST OUTPUT FILE
    *_, srcfidx = extract_offsets(chrtout_files[0], nwm_sources)
    *_, snkfidx = extract_offsets(chrtout_files[0], nwm_sinks)
    if srcfidx.size == 0 and snkfidx.size == 0:
        raise RuntimeError("No features ids matched")

    # replace first row of zeros in vsource and vsink arrays with streamflow data 
    # for time 0 (allows domains with sub-hourly data to average properly)
    # vsource[0, :] = streamflow[source]
    # vsink[0, :] = -streamflow[sink]

    # average streamflow data (if necessary) and replace each row of zeros in vsource and vsink arrays
    times = list(chrtout_groups.keys())
    times = cftime.date2num(times, f'seconds since {times[0].strftime("%Y-%m-%d %H:%M:%S")}')

    source_file = open(args.output/"vsource.th", 'w')
    sink_file = open(args.output/"vsink.th", 'w')

    for row, (dt, files) in enumerate(chrtout_groups.items()):
        print(dt, len(files))
        srcstreamflow = np.zeros(len(srcfidx) + 1, dtype=float)
        snkstreamflow = np.zeros(len(snkfidx) + 1, dtype=float)
        srcstreamflow[0] = snkstreamflow[0] = times[row]
        srcview = srcstreamflow[1:]
        snkview = snkstreamflow[1:]
        for file in files:
            with nc.Dataset(file, 'r') as data:
                srcview += data.variables['streamflow'][srcfidx].filled(0)
                snkview += data.variables['streamflow'][snkfidx].filled(0)
        
        srcview[:] = srcview / len(files)
        snkview[:] = -(snkview / len(files))
        np.savetxt(source_file, np.atleast_2d(srcstreamflow), fmt='%f', delimiter='\t')
        np.savetxt(sink_file, np.atleast_2d(snkstreamflow), fmt='%f', delimiter='\t')
        # vsource[row, :] = srcstreamflow / len(files)
        # vsink[row, :] = -(snkstreamflow / len(files))
    source_file.close()
    sink_file.close()
        

    # with open(os.path.join(out, 'vsource.th'),'w') as src:
    #     np.savetxt(src, np.column_stack([times, vsource]), fmt='%f', delimiter='\t')
        
    # with open(os.path.join(out, 'vsink.th'),'w') as sink:
    #     np.savetxt(sink, np.column_stack([times, vsink]), fmt='%f', delimiter='\t')

    with open(args.output/"source_sink.in", "w") as outfile:
        tmp = list(expand_keys(sources))
        outfile.write(f"{len(tmp)}\n")
        outfile.write("\n".join(tmp))
        outfile.write("\n")
        tmp = list(expand_keys(sinks))
        outfile.write(f"{len(tmp)}\n")
        outfile.write("\n".join(tmp))
        
if __name__ == "__main__":
    args = get_options()
    main(args)


    #    do i=1,nsink   # number of sinks
    #      if (n_sink_source(i)>0) then
    #        do k=1,nt  # number of rows in vsource.th
    #          do j=1,n_sink_source(i)  # n_sink_source => neighboring sources for each sink
    #            isource=i_sink_source(j,i) 
    #            if (vsource(isource,k)>abs(vsink(i,k))) then
    #              vsource(isource,k)=vsource(isource,k)+vsink(i,k)
    #              vsink(i,k)=0d0
    #              exit !no need for finding next neighboring source
    #            else
    #              vsink(i,k)=vsink(i,k)+vsource(isource,k)
    #              vsource(isource,k)=0.0d0
    #            endif
    #          enddo ! j=1,nsource
    #        enddo !k=1,nt
    #      endif
    #    enddo !i=1,nsink
    #    close(33)