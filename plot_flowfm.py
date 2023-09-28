import argparse
import pathlib
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import cftime
import json

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors


def open_csv(filename):
    possible_dates = ["Date (utc)", "Date Time", "Date and Time (GMT)"]
    possible_data = ["gage height (m)", "Water Level", 
                    "Water level (m NAVD88)", "Elevation ocean/est (m NAVD88)", "Prediction"]

    obs_ds = pd.read_csv(filename, low_memory=False)
    # Remove possible spaces in column names
    obs_ds.columns = obs_ds.columns.str.strip()

    # Get date column
    date_col = obs_ds.columns.isin(possible_dates)
    values_col = obs_ds.columns.isin(possible_data)

    if np.count_nonzero(date_col) != 1 and np.count_nonzero(values_col) != 1:
        raise RuntimeError("Unknown CSV format")
    
    date_label = obs_ds.columns[date_col][0]
    value_label = obs_ds.columns[values_col][0]
    df = obs_ds.loc[:, [date_label, value_label]].rename(columns={date_label: "date", value_label: "measurement"})
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)
    df = df.loc[pd.notna(df["date"])]
    df = df.set_index("date").sort_index().tz_localize(None)

    return df


def plot_models(output_dir, station, models, cutter, obs=None, datum=None):
    if not models:
        return
    _tmp = next(iter(models.values()))
    year = _tmp.time[0].values.astype("datetime64[Y]").item().year
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Find timespan that covers all model data
    start = min(pd.Timestamp(d.time.values[0]).tz_localize(None) for d in models.values())
    start = start + cutter
    end = max(pd.Timestamp(d.time.values[-1]).tz_localize(None) for d in models.values())

    # check that start < end
    if end < start:
        print(f"End time {end} is before start time {start}. Try reducing the size of cut")

    if obs is not None:
        obs = obs.loc[start:end]
        if pd.notna(obs.values).any():
            ax.plot(obs.index.to_pydatetime(), obs.values, 'red', label="Measurement", linewidth=2)
    
    colors = ('blue', 'black', 'green', 'purple', 'cyan')
    for c, (f, d) in zip(colors, models.items()):
        d = d.sel({'time': slice(start.tz_localize(None), end.tz_localize(None))})
        if pd.isna(d.values).all():
            # Skip models that are null
            continue
        if len(models) == 1:
            label = "Model"
        else:
            label = f"Model ({f})"
        ax.plot(d.time.values, d.values, color=c, linestyle='--', label=label, linewidth=2)

    ax.legend()
    ax.grid()
    ax.set_title(f"Station ID: {station}", size=20, fontweight='bold')
    ax.set_xlabel(f"Date [{year}]", size=15)
    if datum:
        ax.set_ylabel(f"Water level (m {datum})", size=15)
    else:
        ax.set_ylabel("Water level (m)", size=15)
    plt.savefig(output_dir/f"{station}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)


def detect_model_labels(models):
    while True:
        rv = tuple(m.parent if m.is_file() else m.name for m in models)
        if len(set(rv)) == len(rv):
            return rv
        models = tuple(m.parent for m in models)


def select_schism_nodes(data, indexer, time_indexer=None):
    if time_indexer is None:
        time_indexer = slice(None)
    _elevation = data[time_indexer, indexer].mean(axis=1)
    return _elevation


def select_dflow_sites(data, indexer, time_indexer=None):
    index = data.stations.str.strip().astype(str).values

    if time_indexer is None:
        time_indexer = slice(None)
    waterlevel = data[time_indexer, index == indexer]
    return waterlevel
        

def is_dflow(path):
    try:
        next(path.glob("FlowFM_*_his.nc"))
        return True
    except StopIteration:
        return False


def is_schism(path):
    try:
        next(path.glob("out2d*.nc"))
        return True
    except StopIteration:
        return False
    

def convert_schism_time(times):
    base_date = times.base_date.split()
    rv = cftime.num2date(times.values, f"seconds since {'-'.join(base_date[:3])}")
    return rv.astype('datetime64[ns]')


def read_correspondence_table(path, storms):
    # Correspondence table has the following columns
    # GageID: gages to process
    # Nodes [optional]: SCHSIM nodes to select for GageID
    # ProcessedCSVLoc: Processed csv measurement data
    # Storm: storm identifier
    # Datum: datum identifier
    correspond = pd.read_csv(path, dtype={'GageID': 'string'},
                            converters={'ProcessedCSVLoc': pathlib.Path})
    correspond['GageID'] = correspond['GageID'].str.strip()
    correspond = correspond.set_index('GageID').sort_index()

    if "Storm" in correspond.columns:
        # Filter by storm before uniqueness check
        storm_mask = correspond["Storm"].isin(storms)
        correspond = correspond.loc[storm_mask]
    if not correspond.index.is_unique:
        print(correspond.index[correspond.index.duplicated()])
        raise RuntimeError("GageID needs to be unique")
    if "Datum" in correspond.columns:
        correspond.loc[pd.isna(correspond["Datum"]), "Datum"] = None
    else:
        correspond['Datum'] = None
    return correspond


def main(args):
    cutter = datetime.timedelta(hours=args.cut)

    filehandles = []
    history_files = {}
    for label, d in zip(detect_model_labels(args.model), args.model):
        if is_dflow(d):
            hfile = d/"FlowFM_0000_his.nc"
            fh = xr.open_dataset(hfile)
            filehandles.append(fh)
            fh['station_name'] = fh.station_name.str.strip()
            wl = fh.waterlevel
            wl = wl.set_index(stations='station_name').sortby('stations')
            history_files[(1, label)] = wl
        elif is_schism(d):
            outs = list(d.glob("out2d*.nc"))
            if len(outs) > 1:
                fh = xr.open_mfdataset(outs)
            else:
                fh = xr.open_dataset(outs[0])
            filehandles.append(fh)
            stimes = convert_schism_time(fh['time'])
            fh['time'] = stimes
            wl = fh.elevation
            history_files[(2, label)] = wl
        else:
            raise RuntimeError(f"Unknown model type {d}")

    # load the correspondence table
    correspond = read_correspondence_table(args.correspond, args.storm)

    stations = correspond.index
    for st in stations:
        print("Processing", st)
        metadata = correspond.loc[st]
        obspath = metadata["ProcessedCSVLoc"]
        try:
            obsdata = open_csv(args.obs.joinpath(obspath))
            datum = metadata["Datum"]
        except FileNotFoundError:
            obsdata = None
            datum = None

        model_data = {}
        for (t, f), d in history_files.items():
            if t == 1:
                # Dflow file
                data = select_dflow_sites(d, st, time_indexer=None)
                if data.sizes['stations'] == 0:
                    print("Skipping station with no data", st)
                    continue
                elif data.sizes['stations'] > 1:
                    print("Skipping station because of duplicate data:", st)
                    break
            elif t == 2:
                # SCHISM file
                data = select_schism_nodes(d, json.loads(metadata.loc['Nodes']), time_indexer=None)

            model_data[f] = data    
        else:
            plot_models(args.output, st, model_data, cutter, obs=obsdata, datum=datum)

    # Release resources
    for wl in history_files.values():
        wl.close()
    for fh in filehandles:
        fh.close()


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=pathlib.Path, help="output folder")
    parser.add_argument('--cut', default=12, type=int, help="Number of hours to remove from head of timeseries")
    parser.add_argument("--obs", type=pathlib.Path, help="data folder")
    parser.add_argument("--correspond", type=pathlib.Path, required=True, help='Data correspondence table')
    parser.add_argument("model", nargs='+', type=pathlib.Path, help="model folders")
    parser.add_argument("-s", "--storm", default=["Any"], action="append", help="Storm filter")
    args = parser.parse_args()

    if args.obs and not args.correspond:
        raise RuntimeError("Correspondence table needed to plot observations")

    return args


if __name__ == "__main__":
    args = get_options()
    main(args)