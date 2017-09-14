

CUDA_ENABLED = False


import pylab
import numpy
import wavefront_util
import threading


if CUDA_ENABLED:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    from pyfft.cuda import Plan

    # Initialize CUDA
    cuda.init()

    from pycuda.tools import make_default_context
    global context
    context = make_default_context()
    device = context.get_device()

    def _finish_up():
        global context
        context.pop()
        context = None

        from pycuda.tools import clear_context_caches
        clear_context_caches()

    import atexit
    atexit.register(_finish_up)


    mod = SourceModule("""
#include <cuComplex.h>
#include <math_constants.h>
#include <math.h>


__global__ void multiply_complex64(cuFloatComplex *vec1, cuFloatComplex *vec2, cuFloatComplex *out, const int N)
{
    int id = threadIdx.x+blockIdx.x * blockDim.x;
    out[id] = cuCmulf(vec1[id], vec2[id]);
}

__device__ __forceinline__ cuFloatComplex cexpf(cuFloatComplex z)
{
    cuFloatComplex res;
    float t = expf (z.x);
    sincosf (z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

__global__ void create_propagator(cuFloatComplex * out,
                                  const int width, const int height,
                                  const float lamda_squared, const float kmax,
                                  const float d)
{
    int idx = threadIdx.x+blockIdx.x * blockDim.x;
    int idy = threadIdx.y+blockIdx.y * blockDim.y;

    float fx;
    float fy;
    float sphereSquare;
    cuFloatComplex sphere;
    float exponent_fac;
    cuFloatComplex exponent;
    cuFloatComplex prop;

    fx = idx / (float) width;
    if(idx >= (width+1) / 2) {
        fx -= 1.0f;
    }

    fy = idy / (float) height;
    if(idy >= (height+1) / 2) {
        fy -= 1.0f;
    }

    sphereSquare = 1.0f - lamda_squared * (fx*fx + fy*fy);
    if(sphereSquare >= 0) {
        sphere = make_cuFloatComplex(sqrtf(sphereSquare), 0);
        // The exponential here will be purely imageinary
    }
    else {
        if(d <= 0) {
            sphere = make_cuFloatComplex(0, sqrtf(-sphereSquare));
        }
        else {
            sphere = make_cuFloatComplex(0, -sqrtf(-sphereSquare));
        }
        // The exponential here will be purely real
    }

    exponent_fac = - kmax * d; // 2 * CUDART_PI_F cancelled with kmax
    exponent.x = - sphere.y * exponent_fac;
    exponent.y = sphere.x * exponent_fac;

    prop = cexpf(exponent);

    out[idy * width + idx] = prop;
}



__device__ float3 convert_one_pixel_to_rgb(float h, float s, float v) {
	float r, g, b;

	float f = h/60.0f;
	float hi = floorf(f);
	f = f - hi;
	float p = v * (1 - s);
	float q = v * (1 - s * f);
	float t = v * (1 - s * (1 - f));

	if(hi == 0.0f || hi == 6.0f) {
		r = v;
		g = t;
		b = p;
	} else if(hi == 1.0f) {
		r = q;
		g = v;
		b = p;
	} else if(hi == 2.0f) {
		r = p;
		g = v;
		b = t;
	} else if(hi == 3.0f) {
		r = p;
		g = q;
		b = v;
	} else if(hi == 4.0f) {
		r = t;
		g = p;
		b = v;
	} else {
		r = v;
		g = p;
		b = q;
	}

     return make_float3(r,g,b);
}

__global__ void wavefront_to_rgb(cuFloatComplex *cwavefront, float3 *rgb, const float max_amplitude) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	cuFloatComplex value = cwavefront[id];
     float brightness = sqrtf(value.x * value.x + value.y * value.y) / max_amplitude;
     if(brightness > 1.0) {
        brightness = 1.0;
     }
     float phase = (atan2f(value.y, value.x) / (2 * CUDART_PI) + 0.5) * 360;
	float3 result = convert_one_pixel_to_rgb(phase, 1, brightness);
	rgb[id] = result;
}

  """)
    func_cuda_multiply_complex64 = mod.get_function("multiply_complex64")
    func_cuda_create_propagator = mod.get_function("create_propagator")
    func_cuda_wavefront_to_rgb = mod.get_function("wavefront_to_rgb")


def cuda_multiply(in1, in2, out, size, synchronize=False, threads=64):
    if size % threads != 0:
        raise ValueError("Length must be a power of "+str(threads))
    func_cuda_multiply_complex64(in1, in2, out, numpy.int32(size),
                       block=(threads,1,1), grid=(size / threads, 1))
    if synchronize:
        context.synchronize()

def cuda_multiply_copy(array1, array2):
    array1_gpu = cuda.mem_alloc(array1.nbytes)
    cuda.memcpy_htod(array1_gpu, array1)
    array2_gpu = cuda.mem_alloc(array2.nbytes)
    cuda.memcpy_htod(array2_gpu, array2)

    cuda_multiply(array1_gpu, array2_gpu, array1_gpu, array1.size)
    return array1_gpu


def cuda_ifft2(gpu_array, shape):
    stream = cuda.Stream()
    plan = Plan(shape, stream=stream)
    plan.execute(gpu_array, inverse=True)

def cuda_ifft2_copy(array):
    array_gpu = cuda.mem_alloc(array.nbytes)
    fftg = numpy.empty_like(array)
    cuda.memcpy_htod(array_gpu, array)
    cuda_ifft2(array_gpu, array.shape)
    cuda.memcpy_dtoh(fftg, array_gpu)
    return fftg


def create_propagator_gpu(width, height, lam, d, synchronize=False, gpu_array=None):
    width = numpy.int32(width)
    height = numpy.int32(height)
    lam_squared = numpy.float32(lam*lam)
    kmax = numpy.float32(2*numpy.pi/lam)
    d = numpy.float32(d)

    if gpu_array == None:
        prop_ptr = cuda.mem_alloc(width * height * 2 * 4) # complex64
    else:
        prop_ptr = gpu_array

    threads = 16 # threads in x and y, threads**2 per block
    if width % threads != 0 or height % threads != 0:
        raise ValueError("Length must be a power of "+str(threads))
    func_cuda_create_propagator(prop_ptr, width, height, lam_squared, kmax, d,
        block=(threads,threads,1), grid=(width / threads, height / threads, 1))
    if synchronize:
        context.synchronize()

    return prop_ptr


def create_propagator_cpu(width, height, lam, d):
    kmax = 2*numpy.pi/lam
    
    fx, fy = pylab.meshgrid(pylab.fftfreq(width), pylab.fftfreq(height))
    fx = fx + 0*1j
    fy = fy + 0*1j
    
    propsphere = (1.0 - lam**2 * fx*fx - lam**2 * fy*fy)**.5
    if d >0:
        propsphere -= 2*1j*pylab.imag(propsphere)
        
    prop = pylab.exp(-1j*2*numpy.pi * kmax * d * propsphere).astype(numpy.complex64)
    return prop


def wavefront_to_rgb(wavefront_ptr, rgb_ptr, max_amplitude, size, synchronize=False, threads=64):
    max_amplitude = numpy.float32(max_amplitude)
    if size % threads != 0:
        raise ValueError("Length must be a power of "+str(threads))
    func_cuda_wavefront_to_rgb(wavefront_ptr, rgb_ptr, max_amplitude,
                       block=(threads,1,1), grid=(size / threads, 1))
    if synchronize:
        context.synchronize()


    
    
#def to_color(amplitude, phase, max_amplitude=1.0):
#    phase_bounds = (phase / (2 * numpy.pi)) + 0.5
#    amplitude_bounds = amplitude / float(max_amplitude)
#    rgb = colorsys.hsv_to_rgb(phase_bounds, 1.0, min(amplitude_bounds, 1.0))
#    return rgb
#
#def draw_wavefront(wavefront, max_amplmitude):
#    z_amplitude = numpy.abs(wavefront)
#    z_phase = numpy.angle(wavefront)
#    width, height = wavefront.shape
#    
#    image = numpy.empty((width, height, 3))
#    for y in xrange(height):
#        for x in xrange(width):
#            image[x,y,:] = to_color(z_amplitude[x,y], z_phase[x,y], max_amplitude=max_amplmitude)
#    return image


class PointSourcePropagator:
# TODO: overall strength of the emitter
    def __init__(self, source_x, source_y, initial_phase_complex, wavefront_shape, dr, wavelength, use_cuda=True, wavefront_z=0, source_size=0.05):
        self.source_x = source_x
        self.source_y = source_y
        self.initial_phase = initial_phase_complex * source_size * 50
        self.wavefront_shape = wavefront_shape
        self.dr = dr
        self.wavelength = wavelength
        self.use_cuda = use_cuda
        self.wavefront_z = wavefront_z
        self.last_wavefront = None

        grid_x, grid_y = numpy.meshgrid(numpy.arange(wavefront_shape[1]), numpy.arange(wavefront_shape[0]))
        self.distance2d_squared = ((grid_x - source_x)*dr) ** 2 + ((grid_y - source_y)*dr) ** 2

    def get_dr(self):
        return self.dr

    def get_wavefront_z(self):
        return self.wavefront_z

    def get_wavelength(self):
        return self.wavelength

    def set_use_cuda(self, use_cuda=True):
        self.use_cuda = use_cuda

    def get_use_cuda(self):
        return self.use_cuda

    def propagate(self, target_z):
        d = numpy.sqrt(self.distance2d_squared + (target_z-self.wavefront_z)**2)
        self.last_wavefront = self.initial_phase * numpy.exp(1j*d / self.wavelength) / d
        return self.last_wavefront

    def last_to_rgb(self, brightness=1.0):
        pass

    def get_propagated_wavefront(self):
        return self.last_wavefront



class Propagator:

    def __init__(self, wavefront, dr, wavelength, use_cuda=True, wavefront_z=0):
        wavefront = wavefront.astype(numpy.complex64)
        self.wavefront = wavefront
        self.dr = dr
        self.wavelength = wavelength
        self.use_cuda = use_cuda
        self.wavefront_z = wavefront_z
        
        # fft2 stores the result in a fortran-like array
        wavefront_fft_fortran = numpy.fft.fft2(wavefront)
        self.wavefront_fft = numpy.ascontiguousarray(wavefront_fft_fortran).astype(numpy.complex64)
        
        self.last_wavefront = self.wavefront
        
        if CUDA_ENABLED:
            self.wavefront_fft_gpu = cuda.mem_alloc(self.wavefront_fft.nbytes)
            cuda.memcpy_htod(self.wavefront_fft_gpu, self.wavefront_fft)
            self.out_gpu = cuda.mem_alloc(wavefront.nbytes)
            
            self.temp_gpu = cuda.mem_alloc(self.wavefront.nbytes)
            self.image_gpu = None
    
    def get_dr(self):
        return self.dr

    def get_wavefront_z(self):
        return self.wavefront_z

    def get_wavelength(self):
        return self.wavelength

    def set_use_cuda(self, use_cuda=True):
        self.use_cuda = use_cuda

    def get_use_cuda(self):
        return self.use_cuda


    def propagate(self, target_z):
        d = target_z - self.wavefront_z
        if self.use_cuda:
            return self.__propagate_cuda(d)
        else:
            return self.__propagate_cpu(d)
    
    def __propagate_cuda(self, d):
        d = d / self.dr
        lam = self.wavelength / self.dr
        height, width = self.wavefront_fft.shape
        
        create_propagator_gpu(width, height, lam, d, gpu_array=self.temp_gpu)
        
        # Multiply prop * wavefront_fft
        cuda_multiply(self.temp_gpu, self.wavefront_fft_gpu, self.temp_gpu, self.wavefront.size)
        
        # perform IFFT
        cuda_ifft2(self.temp_gpu, self.wavefront.shape)
        
        # copy to CPU
        out = numpy.empty_like(self.wavefront)
        cuda.memcpy_dtoh(out, self.temp_gpu)
        
        self.last_wavefront = out
        return out
    
    
    def __propagate_cpu(self, d):
        d = d / self.dr
        lam = self.wavelength / self.dr
        height, width = self.wavefront_fft.shape
        
        fx, fy = pylab.meshgrid(pylab.fftfreq(width), pylab.fftfreq(height))
        fx = fx + 0*1j
        fy = fy + 0*1j
        
        ksphere = numpy.sqrt(1.0 - lam*lam * (fx*fx + fy*fy))
        if d >0:
            ksphere = numpy.conjugate(ksphere)
            
        prop = pylab.exp(-1j*2*numpy.pi * d / lam * ksphere)
        out_fft = prop * self.wavefront_fft
        out = numpy.fft.ifft2(out_fft)
        
        self.last_wavefront = out
        return out
    

    
    def last_to_rgb(self, brightness=1.0):
        max_amplitude = numpy.max(abs(self.last_wavefront))
        
        if self.use_cuda:
            if self.image_gpu is None:
                self.image_gpu = cuda.mem_alloc(self.wavefront.size * 3 * 4) # 3 floats
            wavefront_to_rgb(self.temp_gpu, self.image_gpu, max_amplitude / brightness, self.wavefront.size)
            image = numpy.empty((self.wavefront.shape[0], self.wavefront.shape[1], 3),
                                dtype=numpy.float32, order='C')
            cuda.memcpy_dtoh(image, self.image_gpu)
            return image
        else:
            image = wavefront_util.draw_wavefront(self.last_wavefront, max_amplitude / brightness)
            return image

    def get_propagated_wavefront(self):
        return self.last_wavefront


def propagate_array(propagators, target_z, multithreaded=False):
    if multithreaded:
        threads = []

        for propagator in propagators:
            thread = threading.Thread(target=propagator.propagate, args=(target_z,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        wavefronts = [p.get_propagated_wavefront() for p in propagators]
    else:
        wavefronts = [p.propagate(target_z) for p in propagators]
    return wavefronts




#source = create_gaussian_sphere(512, 256)
#prop = Propagation(source, 1.0, 5.0)
#
#ds = numpy.linspace(-20,20,21)#propagation distance (px)
#prefix = "propagation_output/"
#
#start_time = time.time()
#for di,d in enumerate(ds):
#    print "Calculating wavefront at d =",d
#    out = prop.propagate_gpu(d)
#
#    pylab.figure()
#    pylab.imshow(pylab.absolute(out))
#    pylab.xticks([])
#    pylab.yticks([])
#    pylab.title("d=%d"%(d,))
#    pylab.savefig(prefix + "%d_propagation_%d_px.png"%(di,d))
#    pylab.close()
#
#print "Finished in ", time.time()-start_time, "seconds"