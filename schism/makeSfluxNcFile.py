#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 01:26:35 2022

@author: Camaron.George
"""
import os
import time
import pathlib
import pandas as pd
import numpy as np
import netCDF4 as nc
import pyproj as pj
from scipy.spatial import Delaunay
import shapely
import argparse


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=pathlib.Path, help="Output folder")
    parser.add_argument('grid', type=pathlib.Path, help='Gr3 grid')
    parser.add_argument('atm_files', type=pathlib.Path, help='Path to atmospheric files')

    bounds = parser.add_argument_group("bounds")
    bounds.add_argument("--llx", type=float, help="Lower left x coordinate")
    bounds.add_argument("--lly", type=float, help="Lower left y coordinate")
    bounds.add_argument("--urx", type=float, help="Upper right x coordinate")
    bounds.add_argument("--ury", type=float, help="Upper right y coordinate")

    args = parser.parse_args()
    if not args.output.exists():
        args.output.mkdir()
    return args

#this script requires hgrid.utm, hgrid.ll, and netcdf precipitation file
#this path should include all of those files; if it doesn't, paths will need to be adjusted below
path = pathlib.Path('/scratch2/NCEPDEV/ohd/Ryan.Grout/Domains/LakeChamplain')
#path = pathlib.Path('./LC')
gr3 = path/"LC_grid.gr3"

#atmoFile = path/'GEOGRID_LDASOUT_Spatial_Metadata_1km_NWMv2.1.nc'
#outfile = pathlib.Path('/scratch2/NCEPDEV/ohd/Ryan.Grout', 'sflux2sourceInput.nc')
outfile = path/"sflux2sourceInput.nc"

# #read in hgrid.utm to get x,y,z in meters and list of elements

def buffer_to_dict(path, return_boundaries=True):
    from collections import defaultdict
    def first_el(line):
        return line.split(None, 1)[0].strip()

    buf = open(path, 'r')
    description = next(buf)
    rv = {'description': description}
    NE, NP = map(int, next(buf).split())
    nodes = np.loadtxt(buf, max_rows=NP)
    columns = ['x', 'y', 'z'][:nodes.shape[1] - 1]
    rv['nodes'] = pd.DataFrame(nodes[:, 1:], index=pd.Index(nodes[:,0], dtype=int), columns=columns)
    rv['nodes'] = rv['nodes'].sort_index()

    elements = np.loadtxt(buf, dtype=int, max_rows=NE)
    rv['elements'] = pd.DataFrame(elements[:, 1:], index=pd.Index(elements[:, 0], dtype=int))
    rv['elements'] = rv['elements'].sort_index()

    if not return_boundaries:
        return rv
   # Assume EOF if NOPE is empty.
    try:
        NOPE = int(next(buf).split()[0])
    except IndexError:
        return rv
    # let NOPE=-1 mean an ellipsoidal-mesh
    # reassigning NOPE to 0 until further implementation is applied.
    boundaries = defaultdict(dict)
    _bnd_id = 0
    next(buf)
    while _bnd_id < NOPE:
        NETA = int(next(buf).split()[0])
        _cnt = 0
        boundaries[None][_bnd_id] = ind = []
        while _cnt < NETA:
            ind.append(int(first_el(next(buf))))
            _cnt += 1
        _bnd_id += 1
    NBOU = int(next(buf).split()[0])
    _nbnd_cnt = 0
    buf.readline()
    while _nbnd_cnt < NBOU:
        npts, ibtype = map(int, next(buf).split()[:2])
        _pnt_cnt = 0
        if ibtype not in boundaries:
            _bnd_id = 0
        else:
            _bnd_id = len(boundaries[ibtype])
        boundaries[ibtype][_bnd_id] = ind = []
        while _pnt_cnt < npts:
            line = next(buf).split()
            if len(line) == 1:
                ind.append(int(line[0]))
            else:
                index_construct = []
                for val in line:
                    if '.' in val:
                        continue
                    index_construct.append(val)
                ind.append(list(map(int, index_construct)))
            _pnt_cnt += 1
        _nbnd_cnt += 1
    buf.close()
    rv['boundaries'] = dict(boundaries)
    return rv

def get_element_coords(grid, elements=None):
    el = grid['elements'].iloc[:, 1:].stack().to_frame().sort_index()
    if elements:
        el = el.loc[elements]
    XY = grid['nodes'].loc[el[0].values, ['x', 'y']].values.reshape(-1, 3, 2)
    return XY

def transform_arr(transform):
    def transformer(arr):
        return np.column_stack(transform.transform(arr[:, 0], arr[:, 1]))
    return transformer

def main(args):
    
    print("Reading", args.grid)
    start = time.perf_counter()
    grid = buffer_to_dict(args.grid, return_boundaries=False)
    print(time.perf_counter() - start)

    print("Computing area of elements...")
    start = time.perf_counter()
    XY = get_element_coords(grid)
    minx, maxx = XY[:, 0, 0].min(), XY[:, 0, 0].max()
    miny, maxy = XY[:, 0, 1].min(), XY[:, 0, 1].max()
    polys = shapely.polygons(XY)
    del XY
    # Transform polygon coordinates from lat/lon to m
    ee = pj.CRS(8857)
    xform = pj.Transformer.from_crs(ee.geodetic_crs, ee, always_xy=True)
    mpolys = shapely.transform(polys, transform_arr(xform))

    #calculate the area of each element and multiply by 1/density of water for use later
    precip2flux = shapely.area(mpolys)/1000
    del mpolys
    print(time.perf_counter() - start)

    #read in hgrid.ll to get x,y in degrees
    print("Computing centroid of elements...")
    start = time.perf_counter()
    #get bounding box of polys
    #minx, miny, maxx, maxy = shapely.total_bounds(polys)
    propPoints = shapely.get_coordinates(shapely.centroid(polys))
    avgX = propPoints[:, 0]
    avgY = propPoints[:, 1]
    print(time.perf_counter() - start)
    del polys

    #convert NWM locations to lon/lat
    print("Transforming lat/lon...")  
    start = time.perf_counter()

    # Read first atmospheric file for grid spacing
    _file = next(args.atm_files.glob("*.nc4"))
    with nc.Dataset(_file) as tmp:
        y = np.asarray(tmp.variables['latitude'])
        x = np.asarray(tmp.variables['longitude'])
        xstep = x[1] - x[0]
        ystep = y[1] - y[0]
    #step = 0.008333
    #y = np.arange(20, 55, step)
    #x = np.arange(-130, -60, step)
    if not (args.lly is None
            or args.llx is None
            or args.ury is None
            or args.urx is None):
        minx = args.llx
        miny = args.lly
        maxx = args.urx
        maxy = args.ury
    y = y[(y > miny - ystep) & (y < maxy + ystep)]
    x = x[(x > minx - xstep) & (x < maxx + xstep)]

    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.reshape(-1) , Y.reshape(-1)])
    print(time.perf_counter() - start)

    #perform delaunay triangularion on preciptiation points and file the triangles that we need precip data for    
    print("Delauney triangulation...")
    start = time.perf_counter()
    t = Delaunay(points)
    simplex = t.find_simplex(propPoints)
    sim_mask = simplex >= 0
    if (simplex < 0).any():
        print("Invalid centers")
        simplices = t.simplices[simplex[sim_mask]]
    else:
        simplices = t.simplices[simplex]

    # Compute the area of the triangles in delauney triangulation
    area = shapely.area(shapely.polygons(points[simplices]))
    area_cor = np.zeros(simplices.shape)
    polys = np.zeros((len(simplices), 3, 2))
    polys[:, 0, 0] = avgX[sim_mask]
    polys[:, 0, 1] = avgY[sim_mask]
    col_order = [(1, 2), (2, 0), (1, 0)]
    for k, (k1, k2) in enumerate(col_order):
        print(k1, k2)
        polys[:, [1, 2], :] = points[simplices[:,[k1, k2]]]
        area_cor[:, k] = shapely.area(shapely.polygons(polys))
    del polys
    area_cor = area_cor / area.reshape(-1, 1)
    if (area_cor > 1).any() or (area_cor < 0).any():
        print("Bad area cor")

    print(time.perf_counter() - start)
    # open a netCDF file to write
    out_fn = args.output/"sflux2sourceInput.nc"
    print("Writing output", out_fn)
    ncout = nc.Dataset(out_fn,'w',format='NETCDF4')

    # define axis size
    ncout.createDimension('cols',3)
    ncout.createDimension('elem',len(simplices))
    ncout.createDimension('nwmNodesY',len(y))
    ncout.createDimension('nwmNodesX',len(x))

    # create variables
    ncp2f = ncout.createVariable('precip2flux','f8',('elem',))
    ncsim = ncout.createVariable('simplex','int',('elem','cols',))
    nccor = ncout.createVariable('area_cor','f8',('elem','cols',))
    ncavgx = ncout.createVariable('avgX','f8',('elem',))
    ncavgy = ncout.createVariable('avgY','f8',('elem',))
    ncx = ncout.createVariable('x','f8',('nwmNodesY','nwmNodesX'))
    ncy = ncout.createVariable('y','f8',('nwmNodesY','nwmNodesX'))

    # copy axis from original dataset
    ncp2f[:] = precip2flux
    ncsim[:] = simplices
    nccor[:] = area_cor
    ncx[:] = X
    ncy[:] = Y

    ncout.close()

if __name__ == "__main__":
    args = get_options()
    main(args)