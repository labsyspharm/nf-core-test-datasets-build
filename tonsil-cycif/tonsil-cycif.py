# Build a small "raw" CycIF dataset by subsetting real data: Take three cycles
# of a real CyCIF experiment, pull one matching tile from each, and extract a
# grid of small overlapped + jittered windows. The result is saved as
# multi-image OME-TIFF files, one file per cycle.

import numpy as np
import ome_types
import os
import pathlib
import scipy.ndimage as ndi
import subprocess
import sys
import tifffile
import uuid

orig_base = "ImStor/sorger/data/RareCyte/JL503_JERRY/137-TNP_PILOT1-2019OCT/00-Raw_Images"
if len(sys.argv) == 2:
    base = pathlib.Path(sys.argv[1])
else:
    base = pathlib.Path("/n/files") / orig_base

rel_paths = [
    "TNP_PILOT1-20191024/TONSIL2-2D1@20191024_154007_489090/TONSIL2-2D1@20191024_154007_489090.rcpnl",
    "TNP_PILOT1-20191025/TONSIL2-2D1@20191025_144936_975871/TONSIL2-2D1@20191025_144936_975871.rcpnl",
    "TNP_PILOT1-20191028/TONSIL2-2D1@20191028_144750_425244/TONSIL2-2D1@20191028_144750_425244.rcpnl",
]
missing = False
for p in rel_paths:
    in_path = base / p
    if not in_path.exists():
        missing = True
        print(f"MISSING: {p}")
if missing:
    print()
    print("One or more input files could not be located under the following path:")
    print(f"  {base}")
    print("Please specify the location of the TNP_PILOT1-2019* subfolders as")
    print("the sole argument to this script. The last known location was:")
    print(f".../{orig_base}")
    sys.exit(1)

pixel_size = 0.65
ex_wavelengths = [395, 485, 555, 651]
em_wavelengths = [395, 485, 555, 651]
nc = len(ex_wavelengths)
series = 67
x1, y1 = 600, 5
tw, th = 220, 180
nx, ny = 3, 3
ov = 36
ms = 5
torder = 6, 7, 8, 5, 4, 3, 0, 1, 2
assert x1 >= ms and y1 >= ms
assert len(torder) == nx * ny
bfconvert_path = "/home/jmuhlich/development/bioformats/tools/bfconvert"

xs = np.linspace(x1 - ms, x1 - ms + (tw - ov) * (nx - 1), nx, dtype=int)
ys = np.linspace(y1 - ms, y1 - ms + (th - ov) * (ny - 1), ny, dtype=int)
tpos = np.dstack(np.meshgrid(xs, ys)).reshape((len(torder), -1))[torder, ::-1]

rand = np.random.RandomState(0)
ome_base = ome_types.model.OME()
pixels_base = ome_types.model.Pixels(
    dimension_order="XYCZT",
    physical_size_x=pixel_size,
    physical_size_x_unit="µm",
    physical_size_y=pixel_size,
    physical_size_y_unit="µm",
    size_c=nc,
    size_t=1,
    size_x=tw,
    size_y=th,
    size_z=1,
    type="uint16",
)
tpos_flip_y = tpos * [-1, 1] + [tpos[:, 0].max(), 0]
for y, x in tpos_flip_y:
    y *= pixel_size
    x *= pixel_size
    pixels = pixels_base.copy()
    pixels.channels = [
        ome_types.model.Channel(excitation_wavelenth=ex, emission_wavelength=em)
        for ex, em in zip(ex_wavelengths, em_wavelengths)
    ]
    pixels.tiff_data_blocks = [ome_types.model.TiffData(ifd=0, plane_count=nc)]
    pixels.planes = [
        ome_types.model.Plane(
            the_c=c,
            the_t=0,
            the_z=0,
            position_x=x,
            position_x_unit="µm",
            position_y=y,
            position_y_unit="µm",
        )
        for c in range(nc)
    ]
    ome_base.images.append(ome_types.model.Image(pixels=pixels))

for i, p in enumerate(rel_paths, 1):
    in_path = base / p
    out_path = f"TONSIL2-2D1-Cycle{i}.ome.tif"
    print(f"{i}. {in_path.name} => {out_path}")
    print("==========")
    subprocess.run([bfconvert_path, in_path, "tmp.tif", "-series", str(series), "-overwrite"])
    img = tifffile.imread("tmp.tif")
    os.remove("tmp.tif")
    ome = ome_base.copy()
    ome.uuid = uuid.uuid4().urn
    for im, ifd in zip(ome.images, np.arange(0, len(ome.images)) * nc):
        im.pixels.tiff_data_blocks[0].ifd = ifd
    xml = ome.to_xml().encode()
    with tifffile.TiffWriter(out_path, ome=False, shaped=False) as writer:
        for ti, (y, x) in enumerate(tpos):
            shift = np.hstack([0, rand.uniform(-ms, ms, size=2)])
            tile = img[:, y : y + th + ms * 2, x : x + tw + ms * 2]
            tile = ndi.shift(tile, shift)[:, ms:-ms, ms:-ms]
            description = xml if ti == 0 else None
            writer.write(
                tile,
                description=description,
                photometric="MINISBLACK",
                compression="ADOBE_DEFLATE",
                compressionargs={"level": 9},
                predictor="HORIZONTAL",
            )
    print()
    print()
