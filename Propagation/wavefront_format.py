import numpy
import colorsys


class Wavefront:
    def __init__(self, frequencies, indices, x, y, values):
        self.frequencies = frequencies # 1D-row
        self.indices = indices # 1D-array
        self.x = x # 1D-array
        self.y = y # 1D-array
        self.values = values # complex128 2D-array with len(frequencies) columns

    def add_point(self, index, x, y, complex_amplitudes):
        self.indices.append(index)
        self.x.append(x)
        self.y.append(y)
        self.values.append(complex_amplitudes)

    def slice(self, start, end):
        return Wavefront(self.frequencies, self.indices[start:end], self.x[start:end], self.y[start:end],
                         self.values[start:end])

    def get_frequencies(self):
        return self.frequencies

    def frequency_count(self):
        return len(self.frequencies)

    def size(self):
        return len(self.indices)

    def get_raw_index(self, index):
        return self.indices[index]

    def list_raw_indices(self):
        return self.indices

    def find_index(self, raw_index):
        return numpy.where(self.indices == raw_index)[0]

    def get_x(self, index):
        return self.x[index]

    def list_x(self):
        return self.x

    def get_y(self, index):
        return self.y[index]

    def list_y(self):
        return self.y

    def list_xy(self):
        xy = numpy.dstack((self.x, self.y))[0]
        return xy

    def get_value(self, index, frequency_index):
        return self.values[index, frequency_index]

    def list_values(self, frequency_index):
        return self.values[:, frequency_index]

    def list_value_sums(self):
        return numpy.sum(self.values, axis=1)

    def list_value_root_mean_squares(self):
        return numpy.sqrt(abs(numpy.sum(self.values * self.values.conjugate(), axis=1)))


def load_wavefront(path):
    table = numpy.loadtxt(path, skiprows=1, delimiter=",")
    frequencies = table[0, 3::2]
    indices = table[1:, 0]
    x = table[1:, 1]
    y = table[1:, 2]
    if table.dtype == numpy.float64:
        values = numpy.array(table[1:, 3:]).view(dtype=numpy.complex128)
    else:
        raise ValueError("Table must be of type float64")
    return Wavefront(frequencies, indices, x, y, values)


def save_wavefront(wavefront, path):
    table = numpy.zeros((len(wavefront.x)+1, len(wavefront.frequencies)*2+3))
    table[0, 3::2] = wavefront.frequencies
    table[0, 4::2] = wavefront.frequencies
    table[1:,0] = wavefront.indices
    table[1:,1] = wavefront.x
    table[1:,2] = wavefront.y
    real_imag_array = numpy.array(wavefront.values).view(dtype=numpy.float64)
    table[1:,3:] = real_imag_array
    header="Index,X,Y"
    for i in range(len(wavefront.frequencies)):
        header += ",Re(frequency "+str(i)+"),Im(frequency "+str(i)+")"
    numpy.savetxt(path, table, delimiter=",",header=header)


def empty_wavefront(frequencies):
    return Wavefront(frequencies, [], [], [], [])