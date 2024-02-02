"""
Locate and classify elements in a SCHSIM mesh as either a source or sink.

This code is still considered experimental.
"""

import argparse
import pathlib
import json
import csv
import shapely
import numpy as np
import geopandas as gpd
from collections import defaultdict


class Int64Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("gr3", type=pathlib.Path, help="Gr3 grid")
    parser.add_argument("nwm_geom", type=pathlib.Path, help="NWM geometries")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Output json")
    parser.add_argument("--crs", help="CRS of gr3 file. NWM geometries will be converted to this CRS")

    args = parser.parse_args()
    return args

def domain_contains(mesh, boundary_nodes):
    bnd_line = boundary_line(mesh, boundary_nodes)
    bnd_1nodes = get_boundary_elements(mesh, boundary_nodes=boundary_nodes, n=1)
    bnd_2nodes = get_boundary_elements(mesh, boundary_nodes=boundary_nodes, n=2)
    els = list(bnd_1nodes.values())

    # test which side of boundary elements are on
    right = shapely.LinearRing([bnd_line.coords[0]])

    def is_source(intersection_pt, nwm_geom, element, eps=2e-6):
        bdist = bnd_line.line_locate_point(intersection_pt, normalized=True)
        idist = nwm_geom.line_locate_point(intersection_pt, normalized=True)

        bpts = shapely.line_interpolate_point(bnd_line, [bdist-eps, bdist+eps], normalized=True)
        tpts = shapely.line_interpolate_point(nwm_geom, [idist-eps, idist+eps], normalized=True)
        t0 = shapely.contains_properly(els, tpts[0]).any()
        t1 = shapely.contains_properly(els, tpts[1]).any()
        t = shapely.contains_properly(bnd_1nodes, tpts) == [False, True]
    return is_source

def is_source(intersection_pt, nwm_geom, element, eps=2e-6):
    idist = nwm_geom.line_locate_point(intersection_pt, normalized=True)
    tpts = shapely.line_interpolate_point(nwm_geom, [idist-eps, idist+eps], normalized=True)
    t = shapely.contains_properly(element, tpts) == [False, True]
    return all(t)

def is_sink(intersection_pt, nwm_geom, element, eps=2e-6):
    idist = nwm_geom.line_locate_point(intersection_pt, normalized=True)
    tpts = shapely.line_interpolate_point(nwm_geom, [idist-eps, idist+eps], normalized=True)
    t = shapely.contains_properly(element, tpts) == [True, False]
    return all(t)


def crossed_shapes(mesh_shapes, boundary, nwm_geom, distance=None):
    """Compute which mesh elements on the boundary_line are intersected
    by NWM streamlines
    
    Returns a dictionary with nwm_geom index mapped to the mesh elements intersected.
    """
    if distance is None:
        distance = np.finfo('float32').eps
    ishapes = defaultdict(list)
    shapes = tuple(mesh_shapes.values())
    shapely.prepare(shapes)
    shapely.prepare(boundary)

    keys = tuple(list(mesh_shapes.keys()))
    print("Intersecting boundary...")
    I = boundary.intersection(nwm_geom)
    I = I.loc[~shapely.is_empty(I)]
    print("Found", len(I), "intersection points")

    for _id, ipt in I.items():
        print("Processing", _id, end='\r')
        nwm_feature = nwm_geom.loc[_id]
        dists = shapely.line_locate_point(nwm_feature, shapely.get_parts(ipt), normalized=True)
        coord = shapely.get_geometry(ipt, dists.argmax())

        K = shapely.dwithin(shapes, coord, distance)
        which = K.nonzero()[0]
        for wi in which:
            source = is_source(coord, nwm_feature, shapes[wi])
            sink = is_sink(coord, nwm_feature, shapes[wi])
            if source and not sink:
                _type = "source"
            elif sink and not source:
                _type = "sink"
            else:
                _type = None
                print(coord, _id, keys[wi])
            ishapes[_id].append({'element': keys[wi], 'type': _type})
    print()
    return dict(ishapes)
    
def get_crossed_nwm_id(crossed, hydrofabric):
    ids = {}
    for s, nwm in crossed.items():
        ids[s] = hydrofabric.loc[nwm].feature_id.item()
    return ids

def run_lengths(x):
    n = x.size
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
    run_starts, = np.nonzero(loc_run_start)

    # find run values
    run_values = x[loc_run_start]

    # find run lengths
    run_lengths = np.diff(run_starts, append=n)
    return run_values, run_lengths

def nodes_to_coords(mesh, nodes):
    """Get coordinates of a set of nodes"""
    coords = []
    for node in nodes:
        coords.append(mesh['nodes'][node][0:2])
    return coords

def element_coords(mesh, element):
    """Get the coordinates of a mesh element"""
    return nodes_to_coords(mesh, mesh['elements'][element][1:])
    
def boundary_line(mesh, boundary_nodes):
    line = shapely.LineString(nodes_to_coords(mesh, boundary_nodes))
    return shapely.line_merge(line)


def nodes_to_elements(mesh, nodes):
    els = np.asarray(list(mesh['elements'].values()))[:, 1:]
    els_mask = np.isin(els, nodes).any(axis=1)
    return els_mask

def are_adjacent(el1, el2):
    for p in range(3):
        v = el1[p]
        # find the vertex in el2
        for q in range(3):
            if v == el2[q]:
                break
        breakpoint()
        vnext = el1[(p+1) % 3]
        vprev = el1[(p-1) % 3]
        qnext = el2[(q+1) % 3]
        qprev = el2[(q-1) % 3]
        if vnext == qnext or vnext == qprev or vprev == qnext or vprev == qprev:
            return True
    else:
        return False
    
def get_boundary_elements(mesh, boundary_nodes, n=2, ordered=False):
    els = np.asarray(list(mesh['elements'].values()))[:, 1:]
    bnd_els = np.isin(els, boundary_nodes).sum(axis=1) >= n
    bnd_idx = np.asarray(list(mesh['elements'].keys()))[bnd_els]
    verts = np.asarray(nodes_to_coords(mesh, els[bnd_els].flat)).reshape(-1, 3, 2)
    shp = shapely.polygons(verts)
    shapely.prepare(shp)
    if ordered:
        elements = {}
        els = els[bnd_els]
        for be in boundary_nodes:
            x = (els == be).any(axis=1)
            for i in x.nonzero()[0]:
                elements[bnd_idx[i]] = shp[i]
    else:
        elements = dict(zip(bnd_idx, shp))
    return elements


def buffer_to_dict2(path):
    def first_el(line):
        return line.split(None, 1)[0].strip()

    buf = open(path, 'r')
    description = next(buf)
    NE, NP = map(int, next(buf).split())
    nodes = {}
    for _ in range(NP):
        line = next(buf).split()
        vals = list(map(float, line[1:]))
        nodes[int(line[0])] = vals
    elements = {}
    for _ in range(NE):
        line = tuple(map(int, next(buf).split()))
        elements[line[0]] = line[1:]
    
   # Assume EOF if NOPE is empty.
    try:
        #line = next(buf)
        #NOPE = list(map(str.strip, line.split("=")))

        NOPE = int(next(buf).split()[0])
    except (IndexError, StopIteration):
        return {'description': description,
                'nodes': nodes,
                'elements': elements}
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
    return {'description': description,
            'nodes': nodes,
            'elements': elements,
            'boundaries': dict(boundaries)}

def main(args):
    grid = buffer_to_dict2(args.gr3)
    bnd = grid['boundaries'][1][0]
    bndline = boundary_line(grid, bnd)
    geoms = gpd.read_file(args.nwm_geom, engine='pyogrio')
    geoms = geoms.set_index('ID').sort_index()
    
    if args.crs:
        print("Converting hydrofabric to", args.crs)
        geoms = geoms.to_crs(args.crs)
    
    global is_source
    bound_elems = get_boundary_elements(grid, bnd)
    is_source = domain_contains(grid, bnd)
    #bnde = gpd.GeoSeries(bound_elems)
    #bnde.to_file('boundary.json', driver='GeoJSON', index=True)
    #bnde.to_file("boundary.shp", index=True)
    crossed = crossed_shapes(bound_elems, bndline, geoms.geometry)

    with open(args.output, 'w') as out:
        writer = csv.DictWriter(out, fieldnames=["NWM", "Type", "SCHISM"])
        writer.writeheader()

        for nwm, v in crossed.items():
            for item in v:
                writer.writerow({'NWM': nwm, "Type": item['type'], "SCHISM": item['element']})

    with open(args.output, 'w') as out:
       json.dump(crossed, out, indent=2, cls=Int64Encoder)

if __name__ == "__main__":
    args = get_options()
    main(args)