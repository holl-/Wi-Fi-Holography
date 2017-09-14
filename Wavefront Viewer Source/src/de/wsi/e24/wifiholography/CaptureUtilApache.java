package de.wsi.e24.wifiholography;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

public class CaptureUtilApache {

	
	public static float[] absRfft(double[] data) {
		Complex[] complexArray = rfftInternalOrder(data);
		float[] values = new float[complexArray.length / 2];
		for(int i = 0; i < values.length; i++) {
			values[i] = (float) complexArray[complexArray.length-i-1].abs();
		}
		return values;
	}
	
	public static float[] abs(Complex[] complexArray) {
		float[] result = new float[complexArray.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = (float) complexArray[i].abs();
		}
		return result;
	}
	
	private static Complex[] rfftInternalOrder(double[] data) {
		return new FastFourierTransformer(DftNormalization.STANDARD)
				.transform(data, TransformType.FORWARD);
	}
	
	public static int nextPowerOfTwo(int value) {
		int res = 1;
		while(res < value) {
		    res *= 2;
		}
		return res;
	}
	
	/**
	 * Performs a fast fourier transform for the given real values.
	 * The output is laid out as in numpy.rfft, the resulting array is half the size of the input array
	 * @param data
	 * @return
	 */
	public static Complex[] rfft(double[] data) {
		// padding
		int length = nextPowerOfTwo(data.length);
		if(length != data.length) {
			double[] padded = new double[length];
			int offset = length-data.length;
			for(int i = 0; i < data.length; i++) {
				padded[i+offset] = data[i];
			}
			data = padded;
		}
		// TODO padding: is frequency to index still correct?
		
		Complex[] rfft = rfftInternalOrder(data);
		Complex[] result = new Complex[rfft.length / 2];
		for(int i = 0; i < result.length; i++) {
			result[i] = rfft[rfft.length-i-1];
		}
		return result;
	}
	
	public static double getFrequencySpacing(int arrayLength, double samplingRate) {
		double maxFreq = samplingRate / 2;
		return maxFreq / arrayLength;
	}
	
	public static double indexToFrequency(int index, int arrayLength, double samplingRate) {
		return (samplingRate / 2) * (index / (double)arrayLength);
	}
	
	public static int frequencyToIndex(double frequency, int arrayLength, double samplingRate) {
		if(!isFrequencyContained(frequency, samplingRate)) throw new IllegalArgumentException("Frequency "+frequency+" cannot be resolved with sampling rate "+samplingRate);
		return (int) Math.round(2 * frequency * arrayLength / samplingRate);
	}
	
	public static boolean isFrequencyContained(double frequency, double samplingRate) {
		return (2 * frequency / samplingRate) < 1;
	}
	
	public static float score(float[] fft, int lowerLimit, int upperLimit) {
		float total = 0;
		float inner = 0;
		for(int i = 0; i < fft.length; i++) {
			float value = fft[i];
			total += value;
			if(i >= lowerLimit && i < upperLimit) inner += value;
		}
		return inner / total;
	}
	
	/**
	 * Compress array "data" into bins of size "bins", averaging over binned entries.
	 * @param data
	 * @param binCount
	 * @return
	 */
	public static Complex[] compress(Complex[] data, int binCount) {
		List<List<Complex>> bins = IntStream.range(0, binCount).mapToObj(i -> new ArrayList<Complex>()).collect(Collectors.toList());
		
		for(int i = 0; i < data.length; i++) {
			int bin = i * binCount / data.length;
			bins.get(bin).add(data[i]);
		}

		Complex[] result = new Complex[binCount];
		for(int i = 0; i < binCount; i++) {
			result[i] = avg(bins.get(i));
		}
		
		return result;
	}
	
	private static Complex avg(List<Complex> list) {
		Complex result = new Complex(0);
		for(Complex c : list) {
			result = result.add(c);
		}
		return result;
	}
	
	public static Complex[] slice(Complex[] array, int lowerLimit, int upperLimit) {
		Complex[] result = new Complex[upperLimit-lowerLimit];
		for(int i = lowerLimit; i < upperLimit; i++) {
			result[i-lowerLimit] = array[i];
		}
		return result;
	}
}
