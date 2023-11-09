import argparse
import pathlib
from collections import namedtuple

from pysheds.grid import Grid
from RiverMapper.make_river_map import make_river_map

import geopandas as gpd
import numpy as np


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--threshold", type=float, default=0.01, help="Accumulation threshold")
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

            if col_off < window.width and col_off + w > window.width:
                w = window.width % w
            if row_off < window.height and row_off + h > window.height:
                h = window.height % h
            blocks.append(Window(col_off, row_off, col_off + w + overlap, row_off + h + overlap))
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


def main(args):
    print("extracting river thalwegs from", args.dem)
    branches = extract_river_thalwegs(args.dem, args.threshold)
    thalwegs_out = args.output/"thalwegs.shp"
    branches.to_file(thalwegs_out)

    print("creating river shapes...")
    make_river_map(
        tif_fnames = [str(args.dem)],
        thalweg_shp_fname = str(thalwegs_out),
        output_dir = str(args.output)
    )


if __name__ == "__main__":
    args = get_options()
    main(args)


