import pyvista as pv
from pyvista import _vtk
#from pyvista.plotting.opts import ElementType
import xarray as xr
import numpy as np

import pathlib
import platform
import argparse
import datetime
import cftime
import time


def get_options():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_dir", type=pathlib.Path, help="Data directory")

    plotopts = parser.add_argument_group("plot")
    plotopts.add_argument("--cmap", default='ocean', type='str', help="Colormap")
    plotopts.add_argument("--loop", default=1, type=int, help="Number of times to loop animation")
    plotopts.add_argument("--edges", default=False, action='store_true', help="Show mesh edges")

    movieopts = parser.add_argument_group("video")
    movieopts.add_argument("--fps", default=10, type=int, help="Framerate of video file")
    movieopts.add_argument("--output", type=pathlib.Path, help="Movie output file")

    args = parser.parse_args()
    return args

def pv_padded_array(arr):
    valid = arr >= 0
    nvalid = np.count_nonzero(valid, axis=1)
    rv = np.hstack([nvalid.reshape(-1, 1), arr]).ravel()
    return rv[rv >= 0]

def convert_schism_time(times):
    _date_parts = times.base_date.split()
    base_date = datetime.datetime(*map(int, _date_parts[:3]))
    if len(_date_parts) >= 4:
        base_date += datetime.timedelta(hours=float(_date_parts[3]))
    rv = cftime.num2date(times.values, f"seconds since {base_date.isoformat()}")
    return rv.astype('datetime64[ns]')


def test_call(selected_mesh):
     print(selected_mesh.points)

def load_data(args):
    if args.data_dir.is_dir():
         data_files = args.data_dir.glob("*.nc")
    else:
         data_files = args.data_dir
    ds = xr.open_mfdataset(data_files,
                           chunks={'time': 1}, 
                           coords='minimal', data_vars='minimal',
                           parallel=True)
    if ds['time'].dtype.kind != "M":
            ds["time"] = convert_schism_time(ds["time"])
    
    nodes_x = ds.SCHISM_hgrid_node_x
    nodes_y = ds.SCHISM_hgrid_node_y

    pts = np.column_stack([nodes_x, nodes_y, np.zeros(len(nodes_x))])
    faces = pv_padded_array(ds.SCHISM_hgrid_face_nodes - 1).astype(int)
    surf = pv.PolyData(var_inp=pts, faces=faces)
    return surf, ds

def plot_data(surf, ds, args):
    elevation = ds.elevation
    dryNodes = ds.dryFlagNode.astype(bool)
    wet_elevations = xr.where(dryNodes, np.nan, elevation)
    min_elv = round(wet_elevations.min().compute().item())
    
    surf.point_data['elevation'] = wet_elevations[0]
    surf.set_active_scalars('elevation')

    plotter = pv.Plotter()


    write_output = args.output is not None
    print("write_output", write_output)
    if write_output:
        plotter.open_movie(args.output, framerate=args.fps)

    scale_args = {'vertical': True}
    cmap_args = dict(cmap=args.cmap, clim=[-abs(min_elv), abs(min_elv)])
    plotter.add_mesh(surf, scalars='elevation', 
                     **cmap_args, 
                     lighting=False, 
                     scalar_bar_args=scale_args,
                     show_edges=args.edges)
    plotter.view_xy()
    plotter.enable_terrain_style(mouse_wheel_zooms=True, shift_pans=True)
    animating = False

    def exit_callback(plotter, RenderWindowInteractor, event):
        nonlocal animating
        animating = False
        plotter.close()

    if platform.system() == "Windows":
        # Adding closing window callback
        plotter.iren.add_observer(_vtk.vtkCommand.ExitEvent, 
                                lambda render, event: exit_callback(plotter, render, event))
    if not write_output:
        plotter.show(interactive_update=True)

    time_strs = np.datetime_as_string(ds.elevation.time.values.astype("M8[m]"))
    animating = True
    for L in range(args.loop):
        for i in range(len(time_strs)):
            print(L, i)
            surf.point_data['elevation'][:] = wet_elevations[i]
            plotter.add_text(f"{time_strs[i]}", name='iter-label')

            if not animating:
                return
            
            if write_output:
                plotter.write_frame()
            else:
                plotter.update()
                time.sleep(1/args.fps)
    plotter.close()


def main(args):
     surf, ds = load_data(args)
     plot_data(surf, ds, args)
     ds.close()

if __name__ == "__main__":
    args = get_options()
    main(args)