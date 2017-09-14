import numpy
import colorsys

from scipy.interpolate import griddata

import tum_jet



def to_color(c_value, max_amplitude=1.0, tum=True):
    amplitude = 1  # abs(c_value)
    phase = numpy.angle(c_value)
    phase_bounds = (phase / (2 * numpy.pi)) + 0.5
    amplitude_bounds = amplitude / float(max_amplitude)
    if not tum:
        return colorsys.hsv_to_rgb(phase_bounds, 1.0, min(amplitude_bounds, 1.0))
    else:
        return tum_jet.wheel_interpolate(phase_bounds)

def create_grid(width, height, num_x=-1, num_y=-1):
    if num_x < 0:
        num_x = int(width * 100)
    else:
        num_x = int(round(num_x))
    if num_y < 0:
        num_y = int(height * 100)
    else:
        num_y = int(round(num_y))

    xs = numpy.linspace(0.0, width, num_x)
    ys = numpy.linspace(0.0, height, num_y)
    xx,yy = numpy.meshgrid(xs, ys) # 2D arrays for all x and y values
    xyi = numpy.column_stack((xx.flatten(), yy.flatten())) # each row contains (x,y)
    return xx.shape, xyi

def next_power_of_two(value):
    res = 1
    while res < value:
        res *= 2
    return res



def interpolate_to_raster(wavefront, frequency_index=0, width=-1, height=-1, dr=0.01, fit_power_of_two=True, spacing=0.0):
    if width <= 0:
        width = numpy.max(wavefront.list_x())
    if height <= 0:
        height = numpy.max(wavefront.list_y())

    inner_shape, xyi = create_grid(width,height, num_x = width/dr, num_y = height/dr)

    values = wavefront.list_values(frequency_index)
    values_re = numpy.real(values)
    values_im = numpy.imag(values)

    grid_re = griddata(wavefront.list_xy(), values_re, xyi, method='linear', fill_value=0.0).reshape(inner_shape)
    grid_im = griddata(wavefront.list_xy(), values_im, xyi, method='linear', fill_value=0.0).reshape(inner_shape)
    inner_grid = grid_re + 1j * grid_im
    if not fit_power_of_two:
        return inner_grid, (0,0), inner_grid.shape

    w_px = inner_grid.shape[0]
    h_px = inner_grid.shape[1]
    total_width = next_power_of_two(w_px*(1+spacing))
    total_height = next_power_of_two(h_px*(1+spacing))

    outer_grid = numpy.zeros((total_width, total_height), dtype=numpy.complex64)
    x0 = int(round((total_width - w_px) / 2))
    y0 = int(round((total_height - h_px) / 2))

    outer_grid[x0:x0+w_px, y0:y0+h_px] = inner_grid

    return outer_grid, (x0, y0), inner_grid.shape


def phase_image(complex_2d):
    width, height = complex_2d.shape

    image = numpy.empty((width, height, 3))
    for y in xrange(height):
        for x in xrange(width):
            image[x,y,:] = to_color(complex_2d[x,y])
    return image

def blend_wavefront_array(wavefront_array, method="RMS"):
    wavefronts = numpy.array(wavefront_array)
    if method == "RMS":
        wavefronts = abs(wavefronts) **2
        return numpy.sqrt(numpy.sum(wavefronts, axis=0))
    elif method == "Linear":
        return wavefronts.sum(axis=0)

def sub_wavefronts(w1, w2, method="RMS"):
    if method == "RMS":
        return abs(w1) ** 2 - abs(w2) ** 2
    elif method == "Linear":
        return w1-w2