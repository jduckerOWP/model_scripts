import argparse
import pathlib
import xarray as xr
import pandas as pd

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

MODEL_COLORS = mcolors.BASE_COLORS.copy()
del MODEL_COLORS['w']
del MODEL_COLORS['r']

def open_csv(filename):
    obs_ds = pd.read_csv(filename, low_memory=False)
    # Remove possible spaces in column names
    obs_ds.columns = obs_ds.columns.str.strip()

    if 'gage height (m)' in obs_ds.columns:
        data = usgs_csv(obs_ds)
        if "NAVD88" in filename.name:
            return data, "NAVD88"
        else:
            return data, ""
    elif 'Water Level' in obs_ds.columns:
        return coops_csv(obs_ds), ""
    elif 'Water level (m NAVD88)' in obs_ds.columns:
        return fev_csv(obs_ds), "NAVD88"
    else:
        raise RuntimeError("Unknown CSV format")


def usgs_csv(df):
    df['Date (utc)'] = pd.to_datetime(df['Date (utc)'], infer_datetime_format=True)
    df = df.set_index("Date (utc)").sort_index()
    df.index = df.index.tz_localize(None)
    return df['gage height (m)'].rename("observation")


def coops_csv(df):
    df["Date Time"] = pd.to_datetime(df["Date Time"], infer_datetime_format=True)
    df = df.set_index("Date Time").sort_index()
    df.index = df.index.tz_localize(None)
    return df["Water Level"].rename("observation")


def fev_csv(df):
    df["Date and Time (GMT)"] = pd.to_datetime(df["Date and Time (GMT)"], infer_datetime_format=True)
    df = df.set_index("Date and Time (GMT)").sort_index()
    df.index = df.index.tz_localize(None)
    return df["Water level (m NAVD88)"].rename("observation")


def plot_models(output_dir, models, obs=None):
    if not models:
        return
    _tmp = next(iter(models.values()))
    station = _tmp.stations.item().decode().strip()
    year = _tmp.time[0].values.astype("datetime64[Y]").item().year
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Find timespan that covers all model data
    start = min(pd.Timestamp(d.time.values[0]).tz_localize(None) for d in models.values())
    end = max(pd.Timestamp(d.time.values[-1]).tz_localize(None) for d in models.values())

    if obs is not None:
        obs = obs.loc[start:end]
        if pd.notna(obs.values).any():
            ax.plot(obs.index.to_pydatetime(), obs.values, 'r', label="Measurement", linewidth=2)
    
    colors = iter(MODEL_COLORS.values())
    for f, d in models.items():
        d = d.sel({'time': slice(start.tz_localize(None), end.tz_localize(None))})
        if pd.isna(d.values).all():
            # Skip models that are null
            continue
        if len(models) == 1:
            label = "Model"
        else:
            label = f"Model ({f})"
        ax.plot(d.time.values, d.values, color=next(colors), linestyle='--', label=label, linewidth=2)

    ax.legend()
    ax.grid()
    ax.set_title(f"Station ID: {station}", size=20, fontweight='bold')
    ax.set_xlabel(f"Date [{year}]", size=15)
    ax.set_ylabel("Water level (m)", size=15)
    plt.savefig(output_dir/f"{station}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)


def main(args):
    if args.mplstyle:
        plt.style.use(args.mplstyle)

    filehandles = []
    history_files = {}
    for d in args.model:
        hfile = d / "FlowFM_0000_his.nc"
        if hfile.exists():
            fh = xr.open_dataset(hfile)
            filehandles.append(fh)
            fh['station_name'] = fh.station_name.str.strip()
            wl = fh.waterlevel
            wl = wl.set_index(stations='station_name').sortby('stations')
            history_files[hfile] = wl
        else:
            raise FileNotFoundError(f"Cannot find {hfile}")

    # load the correspondence table
    if args.correspond:
        correspond = pd.read_csv(args.correspond, index_col='GageID', 
                                usecols=['GageID', 'Storm', 'ProcessedCSVLoc'], 
                                converters={'ProcessedCSVLoc': pathlib.Path})
        correspond.index = correspond.index.str.strip()
        correspond = correspond.sort_index()

        # Filter by storm before uniqueness check
        storm_mask = correspond["Storm"].isin(args.storm)
        correspond = correspond.loc[storm_mask]
        if not correspond.index.is_unique:
            print(correspond.index[correspond.index.duplicated()])
            raise RuntimeError("GageID needs to be unique")

    _model = next(iter(history_files.values()))
    stations = _model.stations.values
    for station in stations:
        st = station.decode()
        if args.correspond and st in correspond.index:
            obspath = correspond.loc[st, "ProcessedCSVLoc"]
            try:
                obsdata = open_csv(args.obs.joinpath(obspath))[0]
            except FileNotFoundError:
                obsdata = None
        else:
            obsdata = None

        print("Plotting", st)
        model_data = {}
        for f, d in history_files.items():
            data = d.loc[:, station]
            if data.ndim > 1:
                print("Duplicate stations skipped:", st)
                continue
            model_data[f.parent.parent.name] = data
        plot_models(args.output, model_data, obs=obsdata)

    # Release resources
    for wl in history_files.values():
        wl.close()
    for fh in filehandles:
        fh.close()


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=pathlib.Path, help="output folder")
    parser.add_argument("--obs", type=pathlib.Path, help="data folder")
    parser.add_argument("--correspond", type=pathlib.Path, help='Data correspondence table')
    parser.add_argument("model", nargs='+', type=pathlib.Path, help="model folders")
    parser.add_argument("--mplstyle", help="Matplotlib style for plots")
    parser.add_argument("-s", "--storm", default=["Any"], action="append", help="Storm filter")
    args = parser.parse_args()

    if args.obs and not args.correspond:
        raise RuntimeError("Correspondence table needed to plot observations")

    return args


if __name__ == "__main__":
    args = get_options()
    main(args)