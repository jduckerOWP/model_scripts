import argparse
import pathlib
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


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
    if not df.index.tz:
        df.index = df.index.tz_localize("UTC")
    return df['gage height (m)'].rename("observation")


def coops_csv(df):
    df["Date Time"] = pd.to_datetime(df["Date Time"], infer_datetime_format=True)
    df = df.set_index("Date Time").sort_index()
    if not df.index.tz:
        df.index = df.index.tz_localize("UTC")
    return df["Water Level"].rename("observation")


def fev_csv(df):
    df["Date and Time (GMT)"] = pd.to_datetime(df["Date and Time (GMT)"], infer_datetime_format=True)
    df = df.set_index("Date and Time (GMT)").sort_index()
    if not df.index.tz:
        df.index = df.index.tz_localize("UTC")
    return df["Water level (m NAVD88)"].rename("observation")


def plot_models(output_dir, models, obs=None):
    if not models:
        return
    _tmp = next(iter(models.values()))
    station = _tmp.station_name.item().decode()
    year = _tmp.time[0].values.astype("datetime64[Y]").item().year
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    for f, d in models.items():
        ax.plot_date(d.time.values, d.values, label=f.parent.parent.name)

    if obs:
        ax.plot_date(obs.index.to_pydatetime(), obs.values, label="Measurement")
    
    ax.legend(loc='upper left')
    ax.grid()
    ax.set_title(station, size=20)
    ax.set_xlabel(f"Date [{year}]", size=15)
    ax.set_ylabel("Water level (m)", size=15)
    plt.savefig(output_dir/f"{station}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)


def main(args):
    history_files = {}
    stations = None
    for d in args.model:
        hfile = d / "FlowFM_0000_his.nc"
        if hfile.exists():
            history_files[hfile] = fh = xr.open_dataset(hfile)
            _stations = list(map(bytes.decode, fh.station_name.values))
            if stations:
                if (_stations != stations).any():
                    raise RuntimeError(f"Stations differ in {hfile}")
            else:
                stations = _stations

    # load the correspondence table
    if args.correspond:
        correspond = pd.read_csv(args.correspond, index_col='GageID', 
                                usecols=['GageID', 'ProcessedCSVLoc'], 
                                converters={'ProcessedCSVLoc': pathlib.Path})
        if not correspond.index.is_unique:
            print(correspond.index[correspond.index.duplicated()])
            raise RuntimeError("GageID needs to be unique")

    for i, station in enumerate(stations):
        if args.correspond:
            obspath = correspond.loc[station]
            obsdata = open_csv(obspath)
        else:
            obsdata = None

        model_data = {f: d.waterlevel[:, i] for f,d in history_files.items()}
        plot_models(args.output, model_data, obs=obsdata)


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=pathlib.Path, help="output folder")
    parser.add_argument("--obs", type=pathlib.Path, help="data folder")
    parser.add_argument("--correspond", type=pathlib.Path, help='Data correspondence table')
    parser.add_argument("model", nargs='+', type=pathlib.Path, help="model folders")
    args = parser.parse_args()

    if args.obs and not args.correspond:
        raise RuntimeError("Correspondence table needed to plot observations")

    return args


if __name__ == "__main__":
    args = get_options()
    main(args)