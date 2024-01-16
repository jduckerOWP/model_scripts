import argparse
import pathlib
from collections import namedtuple
from itertools import chain

#from pysheds.grid import Grid
import richdem
import rasterio
from rasterio import transform as riotransform
from rasterio import windows as riowindows
import geopandas as gpd
import shapely
import numpy as np
import numba

try:
    from RiverMapper.make_river_map import make_river_map
except ModuleNotFoundError:
    print("Please install RiverMapper from RiverMeshTools")
    raise


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--threshold", type=float, default=5, help="Accumulation threshold")
    parser.add_argument("--epsilon-fill", action='store_true', default=False, help="Epsilon filling can help sometimes.")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=".", help="Output directory")
    parser.add_argument("dem", type=pathlib.Path, help="DEM to process")

    args = parser.parse_args()
    return args


class Window(namedtuple('Window', ['min_x', 'min_y', 'max_x', 'max_y'])):
    __slots__ = ()
    @property
    def height(self):
        return self.max_y - self.min_y
    @property
    def width(self):
        return self.max_x - self.min_x
    
    @classmethod
    def from_shape(cls, shape):
        height, width = shape
        return cls(0, 0, width, height)
    

def subdivide(window, height, width, overlap=0):
    """Subdivide a window into non-overlapping windows of specified height and width.
    Parameters
    ----------
    window: Window 
        Input window to subdivide.
    height, width: int
        Desired height and width of sub windows. Windows of smaller
        sizes may be returned if height and width do not evenly divide
        the dimensions of the input window.
    Returns
    -------
    list[Window]:
        A list of windows that subdivide input window
    """
    blocks = []
    row_blocks = divmod(window.height, height)
    col_blocks = divmod(window.width, width)
    print(row_blocks, col_blocks)
    for j in range(int(row_blocks[0]) + (row_blocks[1] > 0)): # rows
        for k in range(int(col_blocks[0]) + (col_blocks[1] > 0)):  # cols
            h, w = height, width
            col_off = (w * k) - overlap
            row_off = (h * j) - overlap
            col_off = window.col_off + max(col_off, 0)
            row_off = window.row_off + max(row_off, 0)

            if col_off < window.width and col_off + w > window.width:
                w = window.width % w
            if row_off < window.height and row_off + h > window.height:
                h = window.height % h
            blocks.append(riowindows.Window(col_off, row_off, w + overlap, h + overlap))
            #blocks.append(riowindows.Window(col_off, row_off, col_off + w + overlap, row_off + h + overlap))
    return blocks


def array_bytes(dtype, shape):
    """Get size of array in bytes with specific dtype and shape.
    
    Parameters
    ----------
    dtype: np.dtype, str
        Array dtype
    shape: tuple
        Array shape
        
    Returns
    -------
    int:
        Array size in bytes
    """
    size = np.dtype(dtype).itemsize
    for s in shape:
        size *= s
    return size


def extract_river_thalwegs(dem, threshold):
    grid = Grid.from_raster(str(dem))
    #blocks = subdivide(grid)

    data = grid.read_raster(str(dem)) 
    flooded_dem = grid.fill_pits(data)
    inflated_dem = grid.resolve_flats(flooded_dem)

    fdir = grid.flowdir(inflated_dem)
    acc = grid.accumulation(fdir, apply_output_mask=False)
    acc_threshold = acc.max() * threshold
    branches = grid.extract_river_network(fdir, acc > acc_threshold)
    branches = gpd.GeoDataFrame.from_features(branches).set_crs(grid.crs)
    return branches

@numba.njit
def flow_indegree(flowdir, mask):
    indices = np.nonzero(mask)

    offsets = np.array([[0, -1, 5],
                        [-1, -1, 6],
                        [-1, 0, 7],
                        [-1, 1, 8],
                        [0, 1, 1],
                        [1, 1, 2],
                        [1, 0, 3],
                        [1, -1, 4]])
    rows = flowdir.shape[0]
    cols = flowdir.shape[1]
    indegrees = {}
    for i, j in zip(*indices):
        indeg = 0
        for n in range(8):
            ni, nj, loc = offsets[n]
            ni = i + ni
            nj = j + nj
            if ni < 0 or nj < 0 or ni >= rows or nj >= cols:
                continue
            if not mask[ni, nj]:
                continue
            N = flowdir[ni, nj]
            if N[0] == 0 and N[loc] > 0:
                indeg += 1
        indegrees[(i, j)] = indeg
    return indegrees

def river_lines(indegrees, flowdirs):
    lines = []
    _indegrees = dict(indegrees)
    offsets = np.array([[0, -1],
                    [-1, -1],
                    [-1, 0],
                    [-1, 1],
                    [0, 1],
                    [1, 1],
                    [1, 0],
                    [1, -1]])
    
    # collect origins (indegree==0)
    origin_points = set()
    for k, v in _indegrees.items():
        if v == 0:
            origin_points.add(k)
    while origin_points:
        coords = []
        x, y = origin_points.pop()
        
        while True:
            d = flowdirs[x, y]
            if d[0] == 0:
                receiver = np.argmax(d[1:])
                ex = x + offsets[receiver, 0]
                ey = y + offsets[receiver, 1]
                endpt = (ex, ey)
                coords.append(((x, y), endpt))

                if endpt in _indegrees and _indegrees[endpt] > 0:
                    _indegrees[endpt] -= 1
                    x, y = ex, ey
                else:
                    break
            else:
                break
        lines.append(list(chain.from_iterable(coords)))
    return lines

def transform_to_coords(transform, line):
    coords = zip(*riotransform.xy(transform, *zip(*line)))
    return list(coords)
    
def transform_coords(transform, lines):
    shapes = []
    for L in lines:
        if L:
            shapes.append(shapely.LineString(transform_to_coords(transform, L)))
    return shapes

def extract_rivers(flowdir, scale):

    ind = flow_indegree(flowdir, scale)
    streams = river_lines(ind, flowdir)
    return streams

def main(args):
    print("extracting river thalwegs from", args.dem)

    with rasterio.open(args.dem) as raster:
        window = riowindows.Window(0, 0, 20000, 11999)
        print(raster.width, raster.height)
        DEM = raster.read(1, window=window)
        transform = raster.window_transform(window)
    
        print("DEM size:", DEM.shape)
        rdDEM = richdem.rdarray(DEM, no_data=raster.nodata)
        rdDEM.geotransform = transform.to_gdal()
        richdem.FillDepressions(rdDEM, epsilon=args.epsilon_fill, in_place=True)
        richdem.ResolveFlats(rdDEM, in_place=True)

        print("Flow Direction computation...")
        flow = richdem.FlowProportions(rdDEM, method='D8')
        print("done")

        print("Computing flow accumulation from flow direction...")
        acc = richdem.FlowAccumFromProps(flow)
        accmin = acc.min()
        accmax = acc.max()
        print("Computing scaled accumulation values...")
        np.subtract(acc, accmin, out=acc)
        np.divide(acc, accmax - accmin, out=acc)
        np.multiply(acc, 1023, out=acc)
        np.add(acc, 1, out=acc)
        np.log2(acc, out=acc)
        scales = np.floor(acc, out=acc)
        #scales = np.floor(np.log2(1+1023*((acc - accmin)/(acc.max()-accmin))))
        np.greater_equal(scales, args.threshold, out=scales)
        del acc
        print("done")

    print("extracting rivers")
    streams = extract_rivers(flow, scales)
    del scales
    del flow
    shapes = gpd.GeoDataFrame(geometry=transform_coords(transform, streams))
    thalwegs_out = args.output/"thalwegs.shp"
    shapes.to_file(thalwegs_out)

    print("creating river shapes...")
    make_river_map(
        tif_fnames = [str(args.dem)],
        thalweg_shp_fname = str(thalwegs_out),
        output_dir = str(args.output)
    )


if __name__ == "__main__":
    args = get_options()
    main(args)


