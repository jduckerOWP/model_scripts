import argparse
import pathlib
import numpy as np
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


def plot_models(output_dir, models, obs=None, datum=None):
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
    if datum:
        ax.set_ylabel(f"Water level (m {datum})", size=15)
    else:
        ax.set_ylabel("Water level (m)", size=15)
    plt.savefig(output_dir/f"{station}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)


def main(args):
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
                                converters={'ProcessedCSVLoc': pathlib.Path})
        correspond.index = correspond.index.str.strip()
        correspond = correspond.sort_index()

        # Filter by storm before uniqueness check
        storm_mask = correspond["Storm"].isin(args.storm)
        correspond = correspond.loc[storm_mask]
        if not correspond.index.is_unique:
            print(correspond.index[correspond.index.duplicated()])
            raise RuntimeError("GageID needs to be unique")
        correspond.loc[pd.isna(correspond["Datum"]), "Datum"] = None

    _model = next(iter(history_files.values()))
    stations = _model.stations.values
    for station in stations:
        st = station.decode()
        print("Processing", st)
        if args.correspond and st in correspond.index:
            metadata = correspond.loc[st]
            obspath = metadata["ProcessedCSVLoc"]
            try:
                obsdata = open_csv(args.obs.joinpath(obspath))
                datum = metadata["Datum"]
            except FileNotFoundError:
                obsdata = None
                datum = None
        else:
            obsdata = None
            datum = None

        model_data = {}
        for f, d in history_files.items():
            data = d.loc[:, station]
            if data.ndim > 1:
                print("Skipping station because of duplicate data:", st)
                break
            model_data[f.parent.parent.name] = data
        else:
            plot_models(args.output, model_data, obs=obsdata, datum=datum)

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
    parser.add_argument("-s", "--storm", default=["Any"], action="append", help="Storm filter")
    args = parser.parse_args()

    if args.obs and not args.correspond:
        raise RuntimeError("Correspondence table needed to plot observations")

    return args


if __name__ == "__main__":
    args = get_options()
    main(args)