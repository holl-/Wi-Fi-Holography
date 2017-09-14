package de.wsi.e24.wifiholography;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.complex.Complex;
import org.jtransforms.fft.DoubleFFT_1D;

public class CaptureUtilJTransforms {

	
	public static float[] absRfft(double[] data) {
		inPlaceRFFT(data);
		
		float[] values = new float[data.length / 2];
		for(int i = 1; i < values.length; i++) { // ignore first element (0)
			double real = data[i*2];
			double imag = data[i*2+1];
			values[i] = (float) Math.sqrt(real*real+imag*imag);
		}
		return values;
	}
	
	@Deprecated
	public static float[] abs(double[] complexArray) {
		float[] result = new float[complexArray.length/2];
		for(int i = 0; i < result.length; i++) {
			double real = complexArray[i*2];
			double imag = complexArray[i*2+1];
			result[i] = (float) Math.sqrt(real*real+imag*imag);
		}
		return result;
	}
	
	public static double[] inPlaceRFFT(double[] data) {
		DoubleFFT_1D fft = new DoubleFFT_1D(data.length);
		fft.realForward(data);
		return data;
	}
	
	public static double getFrequencySpacing(int arrayLength, double samplingRate) {
		double maxFreq = samplingRate / 2;
		return maxFreq / arrayLength;
	}
	
	public static double indexToFrequency(int index, int arrayLength, double samplingRate) {
		return (samplingRate / 2) * (index / (double)arrayLength);
	}
	
	public static int frequencyToIndex(double frequency, int arrayLength, double samplingRate) {
		return (int) Math.round(2 * frequency * arrayLength / samplingRate);
	}
	
	public static float score(float[] fft, int lowerLimit, int upperLimit) {
		float total = 0;
		float inner = 0;
		for(int i = 1; i < fft.length; i++) { // ignore index 0
			float value = fft[i];
			total += value;
			if(i >= lowerLimit && i < upperLimit) inner += value;
		}
		return inner / total;
	}
	

	@Deprecated
	public static double[] inPlaceDiv(double[] data1, double[] data2) {
		if(data1.length != data2.length) throw new IllegalArgumentException();
		if(data1.length % 2 != 0) throw new IllegalArgumentException();
		
		for(int i = 0; i < data1.length/2; i++) {
			double r1 = data1[i*2];
			double i1 = data1[i*2+1];
			double r2 = data2[i*2];
			double i2 = data2[i*2+1];
			double denominator = r2*r2 + i2*i2;
			double real = (r1*r2-i1*i2) / denominator;
			double imag = (i1*r2-i2*r1) / denominator;
			data1[i*2] = real;
			data1[i*2+1] = imag;
		}
		return data1;
	}
	
	/**
	 * Compress array "data" into bins of size "bins", averaging over binned entries.
	 * @param data
	 * @param binCount
	 * @return
	 */
	@Deprecated
	public static Complex[] compress(double[] data, int binCount) {
		List<List<Complex>> bins = IntStream.range(0, binCount).mapToObj(i -> new ArrayList<Complex>()).collect(Collectors.toList());
		
		for(int i = 0; i < data.length/2; i++) {
			Complex c = new Complex(data[i*2], data[i*2+1]);
			int bin = i * binCount / data.length;
			bins.get(bin).add(c);
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
