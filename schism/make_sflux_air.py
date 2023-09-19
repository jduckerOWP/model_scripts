"""
Assemble the atmospheric forcings for SCHISM.

This script writes sflux_air*.nc files that are ingested by schism.
"""

import argparse
import pathlib
import itertools
import netCDF4 as nc
import cftime
import xarray as xr

def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter.
    
    Recipe from itertools.
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch

def pre_crop(ds, bounds=None):
    if bounds:
        minx, miny, maxx, maxy = bounds
        ds = ds.sel(latitude=(ds.latitude > miny) & (ds.latitude < maxy),
                        longitude=(ds.longitude > minx) & (ds.longitude < maxx))
    return ds


def write_sflux_air(aorc_files, output_dir, bounds=None, break_every=30):
    tmplate = "sflux_air_1.{:0=4d}.nc"
    counter = itertools.count(1)

    # Open one of the files to discover the dimensions of the cropped area
    with xr.open_dataset(aorc_files[0], engine='netcdf4') as ds:
        ds = pre_crop(ds, bounds=bounds)
        nrows = ds.dims['latitude']
        ncols = ds.dims['longitude']

    for grp in batched(aorc_files, break_every):
        fn = tmplate.format(next(counter))
        print("Creating", fn)

        ncout = nc.Dataset(output_dir/fn, 'w', format='NETCDF4')
        ncout.createDimension('time',len(grp)+1)
        ncout.createDimension('ny_grid',nrows)
        ncout.createDimension('nx_grid',ncols)
        nctime = ncout.createVariable('time','f4',('time',))
        nclon = ncout.createVariable('lon','f4',('ny_grid','nx_grid',))
        nclat = ncout.createVariable('lat','f4',('ny_grid','nx_grid',))
        ncu = ncout.createVariable('uwind','f4',('time','ny_grid','nx_grid',))
        ncv = ncout.createVariable('vwind','f4',('time','ny_grid','nx_grid',))
        ncp = ncout.createVariable('prmsl','f4',('time','ny_grid','nx_grid',))
        nct = ncout.createVariable('stmp','f4',('time','ny_grid','nx_grid',))
        ncq = ncout.createVariable('spfh','f4',('time','ny_grid','nx_grid',))

        times = []
        for i, src in enumerate(grp):
            print(f"{src.name}".ljust(80), end='\r')
            with xr.open_dataset(src, engine='netcdf4') as data:    
                data = pre_crop(data, bounds=bounds)
                data = data.squeeze()
                if i == 0:
                    print("Copying longitude/latitude")
                    nclon[:] = data['longitude'].values.reshape(-1, ncols)
                    nclat[:] = data['latitude'].values.reshape(nrows, -1)
                nct[i] = data['TMP_2maboveground']
                ncq[i] = data['SPFH_2maboveground']
                ncu[i] = data['UGRD_10maboveground']
                ncv[i] = data['VGRD_10maboveground']
                ncp[i] = data['PRES_surface']

                times.append(data['time'].values.astype('datetime64[ms]').item())

        t0 = times[0]
        base_date = [t0.year,
                     t0.month,
                     t0.day,
                     t0.hour]
        ref = f"days since {t0.strftime('%Y-%m-%d %H:%M:%S')}"
        times.append(times[-1] + (times[1] - t0))
        nctime[:] = cftime.date2num(times, ref, calendar='julian')
        
        nctime.long_name = "Time"
        nctime.standard_name = "time"
        nctime.units = ref
        nctime.base_date = base_date
        nclon.long_name = "Longitude"
        nclon.standard_name = "longitude"
        nclon.units = "degrees_east"
        nclat.long_name = "Latitude"
        nclat.standard_name = "latitude"
        nclat.units = "degrees_north"
        ncu.long_name = "Surface Eastward Air Velocity (10m AGL)"
        ncu.standard_name = "eastward_wind"
        ncu.units = "m/s"
        ncv.long_name = "Surface Northward Air Velocity (10m AGL)"
        ncv.standard_name = "northward_wind"
        ncv.units = "m/s"
        ncp.long_name = "Pressure reduced to MSL"
        ncp.standard_name = "air_pressure_at_sea_level"
        ncp.units = "Pa"
        nct.long_name = "Surface Air Temperature (2m AGL)"
        nct.standard_name = "air_temperature"
        nct.units = "K"
        ncq.long_name = "Surface Specific Humidity (2m AGL)"
        ncq.standard_name = "specific_humidity"
        ncq.units = "kg/kg"
        ncout.close()
        

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("sflux", type=pathlib.Path, help="Path to sflux2sourceInput.nc")
    parser.add_argument("nwm_output", type=pathlib.Path, help="Path to NWM output files")
    parser.add_argument("-o", "--output", type=pathlib.Path, default='.', help="Output directory")

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

    # Create output directory if it doesn't exist
    if not args.output.exists():
        args.output.mkdir()
    write_sflux_air(files, args.output, bounds=bounds, break_every=24*30)


if __name__ == "__main__":
    args = get_options()
    main(args)
    
