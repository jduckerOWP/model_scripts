# Model Scripts Repository
Usage info is embedded in each script and can be accessed using `--help` or `-h`.

## dl_data.py -> adjust_gage.py
Download and adjust gage readings from USGS.

Data is downloaded from waterservices.usgs.gov using the JSON API. The currently supported codes are:

00060: streamflow
00065: gage height
00045: precip total
00046: physical precip total
62620: elevation
62615: waterlevel

Example:
```
python dl_data.py sites.txt -p 00065 -p 00060 --output ./Measurement_Data
```

The `adjust_gage.py` script will adjust/correct the downloaded data based on values found in a user-provided table.
The default adjustments are converting ft to m.
The correction file is a CSV with three columns:

* Gage ID
* Datum correction (m) - adjustment to make to the datum value
* Gage datum (m) - adjustment to the gage datum value

The adjustments are cumulative and added to the gage height value after it has been converted to meters.

Example:
```
python adjust_gage.py Corrections.csv ./Measurment_Data ./Adjusted_Data
```

## merge_dflow.py
Concatenate dflow output files along time axis, but in a memory efficient way. This script uses xarray to perform the concatenation.

Example:
```
python merge_dflow.py --output merged_output.nc  output1.nc output2.nc
```

## wl_plotter.py
Plot dflow results and compare with  measurement data.

Example:
```
python wl_plotter.py --correspond Test_Correspondence.csv --storm Any --storm Harvey --obs ./Measurment_Data --output ./OutPlots ./Model_Output
```

To solve for tidal coefficients, you can pass the `--tide` argument to `wl_plotter.py`. This requires pytides to be installed in the environment from https://github.com/groutr/pytides.


## plot_flowfm.py
Plot multiple dflow results compare with each other and optionally with measurement data.

Models are processed in the order they are given to the script.

Example:
```
python plot_flowfm.py --correspond Test_Correspondence.csv --storm Any --storm Harvey --obs ./Measurment_Data --output ./OutPlots ./Model_Output1 ./Model_Output2 ./Model_Output3
```

## Correspondence Table and Measurment Data
The plotting scripts use a correspondence table to map stations to their measurement table. It was decided that having this mapping explicit aided understanding issues and reduced the overall maintenance burden. The correspondence table two columns that map a station to file path and can contain several columns of metadata about the station.

The three *required* columns of the correspondence table are "GageID" and "ProcessedCSVLoc" which map the gage id to the CSV file path.

The columns of a correspondence table are (order agnostic):

* GageID - The id of the gage (in DFlow)
* ProcessedCSVLoc - File path for measurement data
* Storm - If measurements are storm specific. "Any" if storm agnostic
* Datum - Datum of measurement data.

GageID must be a unique key column (a gage id should only appear once unless using storm specific measurments). If the GageID column is not unique after processing, an error will be raised.

### Storm filters
A storm filter can be used to allow multiple gage ids. The pair of (GageID, Storm) must be unique per correspondence table. "Any" is the default storm filter but can be overridden using the `--storm` argument to the plotting scripts.

Example:
```
python wl_plotter.py --storm Florence --storm Any ...
```

## environment.yml
Conda environment specification. The minimum compatible version of each package is listed here.
An environment can be created by executing `conda create -f environment.yml`