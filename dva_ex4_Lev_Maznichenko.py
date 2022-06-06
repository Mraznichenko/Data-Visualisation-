"""
How to run the script:

This script does not need a bokeh server, simnply run it with
´´´
python dva_ex4_skeleton_FS22.py
´´´


Point Distribution:

1 Point: Divergence Plot
1 Point: Vorticity Plot
1.5 Points: Vector Coloring
1.5 Points: Hedgehog Overlays

"""

import numpy as np
import os
import bokeh

from bokeh.layouts import layout, row
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker, ColumnDataSource
from colorcet import CET_L16
from bokeh.layouts import column, row, gridplot

from colorsys import hsv_to_rgb


output_file('DVA_ex4.html')
color = CET_L16

HEDGEHOG_OPACITY = 0.85
HEDGEHOG_GRID_SIZE = 10

def to_bokeh_image(rgba_uint8):
    " Essentially converts an rgba image of uint8 type to a image usable by bokeh "
    if len(rgba_uint8.shape) > 2 \
            and int(bokeh.__version__.split(".")[0]) >= 2 \
            and int(bokeh.__version__.split(".")[1]) >= 2:

        np_img2d = np.zeros((rgba_uint8.shape[0], rgba_uint8.shape[1]), dtype=np.uint32)
        view = np_img2d.view(dtype=np.uint8).reshape(rgba_uint8.shape)
        view[:] = rgba_uint8[:]
    else:
        np_img2d = rgba_uint8
    return [np_img2d]

def get_divergence(vx_wind, vy_wind):
    # Use np.gradient to calculate the gradient of a vector field. Find out what exactly the return values represent and
    # use the appropriate elements for your calculations
    gx = np.gradient(vx_wind)
    gy = np.gradient(vy_wind)
    gx = np.array(gx)
    gy = np.array(gy)
    return np.add(gx[0,:,:,19], gy[1,:,:,19])


def get_vorticity(vx_wind, vy_wind):
    # Calculate the gradient again and use the appropriate results to calculate the vorticity. Think about what happens
    # to the z-component and the derivatives with respect to z for a two dimensional vector field.
    # (You can save the gradient in the divergence calculations or recalculate it here. Since the gradient function is
    # fast and we have rather small data slices the impact of recalculating it is negligible.)

    # your code
    gx = np.gradient(vx_wind)
    gy = np.gradient(vy_wind)
    gx = np.array(gx)
    gy = np.array(gy)
    vort_v = np.subtract(gx[1,:,:,19], gy[1,:,:,19])
    return vort_v

# load and process the required data
print('processing data')
x_wind_file = 'Uf24.bin'
x_wind_path = os.path.abspath(os.path.dirname(x_wind_file))
x_wind_data = np.fromfile(os.path.join(x_wind_path, x_wind_file), dtype=np.dtype('>f'))
x_wind_data = np.reshape(x_wind_data, [500, 500, 100], order='F')
x_wind_data = np.flipud(x_wind_data)

# replace the missing "no data" values with the average of the dataset
filtered_average = np.average(x_wind_data[x_wind_data < 1e35])
x_wind_data[x_wind_data == 1e35] = filtered_average

y_wind_file = 'Vf24.bin'
y_wind_path = os.path.abspath(os.path.dirname(y_wind_file))
y_wind_data = np.fromfile(os.path.join(y_wind_path, y_wind_file), dtype=np.dtype('>f'))
y_wind_data = np.reshape(y_wind_data, [500, 500, 100], order='F')
y_wind_data = np.flipud(y_wind_data)

# replace the missing "no data" values with the average of the dataset
filtered_average = np.average(y_wind_data[y_wind_data < 1e35])
y_wind_data[y_wind_data == 1e35] = filtered_average
wind_divergence = get_divergence(x_wind_data, y_wind_data)
wind_vorticity = get_vorticity(x_wind_data, y_wind_data)
print('data processing completed')
color_mapper_divergence = LinearColorMapper(palette=CET_L16, low=np.amin(wind_divergence),
                                            high=np.amax(wind_divergence))
divergence_plot = figure(title="Divergence", **fig_args)
divergence_plot.image(image=to_bokeh_image(wind_divergence), color_mapper=color_mapper_divergence, **img_args)
divergence_color_bar = ColorBar(color_mapper=color_mapper_divergence, **cb_args)
divergence_plot.add_layout(divergence_color_bar, 'right')

color_mapper_vorticity = LinearColorMapper(palette=CET_L16, low=np.amin(wind_vorticity),
                                            high=np.amax(wind_vorticity))
vorticity_plot = figure(title="Voriticty", **fig_args)
vorticity_plot.image(image=to_bokeh_image(wind_vorticity), color_mapper=color_mapper_vorticity, **img_args)
vorticity_color_bar = ColorBar(color_mapper=color_mapper_vorticity, **cb_args)
vorticity_plot.add_layout(vorticity_color_bar, 'right')

layout = row(divergence_plot, vorticity_plot)
show(layout)

