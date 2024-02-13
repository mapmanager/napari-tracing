import os
import tifffile

import napari
from napari import Viewer

from napari_tracing import TracerWidget

# import 2d sample
# path = os.path.join('data', 'sample-2d.tif')
filename = 'sample-2d.tif'
path = os.path.join('data', filename)

sample_2d = tifffile.imread(path)

# viewer = Viewer()
viewer = napari.view_image(sample_2d, name=os.path.splitext(filename)[0])

napari.run()