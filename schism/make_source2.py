"""
Assemble the precipitation forcings for SCHISM

This script write thes source2.nc file ingested by SCHISM.
"""

import argparse
import pathlib
import numpy as np
import xarray as xr
import netCDF4 as nc


def pre_crop(ds, bounds=None):
    if bounds:
        minx, miny, maxx, maxy = bounds
        lastep = (ds.latitude[1] - ds.latitude[0]).item()
        lostep = (ds.longitude[1] - ds.longitude[0]).item()
        ds = ds.sel(latitude=(ds.latitude > miny - lastep) & (ds.latitude < maxy + lastep),
                        longitude=(ds.longitude > minx - lostep) & (ds.longitude < maxx + lostep))
    return ds


def write_source2_AORC(sflux, aorc_files, output_dir, bounds=None):
    with xr.open_dataset(sflux) as data:
        precip2flux = data['precip2flux'].values
        simplex = data['simplex'].values
        area_cor = data['area_cor'].values

    # Write source2.nc
    out_file = output_dir/"source2.nc"
    ncvout = nc.Dataset(out_file, 'w', format="NETCDF4")
    ncvout.createDimension('time_vsource',len(aorc_files))
    ncvout.createDimension('nsources',len(precip2flux))
    ncvso = ncvout.createVariable('vsource','f8',('time_vsource','nsources',))
    
    tmp = np.ma.zeros_like(simplex, dtype='float64')
    for i, fn in enumerate(aorc_files):
        print(fn, f"{round((i+1)/len(aorc_files)*100, 2)}%", end='\r')
        with xr.open_dataset(fn, engine='netcdf4') as ds:
            ds = pre_crop(ds, bounds=bounds)

            apcp = ds['APCP_surface'].to_masked_array().ravel() / 3600
            tmp[:] = apcp[simplex]
            np.ma.multiply(tmp, area_cor, out=tmp)
            np.ma.sum(tmp, axis=1, out=tmp[:, 0])
            np.ma.multiply(tmp[:, 0], precip2flux, out=tmp[:, 0])
            ncvso[i, :] = tmp[:, 0]
    print()

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("sflux", type=pathlib.Path, help="Path to sflux2sourceInput.nc")
    parser.add_argument("nwm_output", type=pathlib.Path, help="Path to NWM output files")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Output directory")

    args = parser.parse_args()

    # validate files/directories
    if not args.sflux.is_file():
        raise FileNotFoundError(f"{args.sflux} not found")
    if not args.nwm_output.is_dir():
        raise IOError(f"NWM output directory not found: {args.nwm_output}")
    return args


def main(args):
    # get sflux bounds
    with xr.open_dataset(args.sflux) as data:
        minx, maxx = data.x.values.min(), data.x.values.max()
        miny, maxy = data.y.values.min(), data.y.values.max()

    bounds = (minx, miny, maxx, maxy)
    files = sorted(args.nwm_output.glob("*.nc4"))

    write_source2_AORC(args.sflux, files, args.output, bounds=bounds)


if __name__ == "__main__":
    args = get_options()
    main(args)
    