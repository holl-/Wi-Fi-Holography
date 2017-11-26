package de.wsi.e24.wifiholography;

import static de.wsi.e24.wifiholography.CaptureUtilApache.abs;
import static de.wsi.e24.wifiholography.CaptureUtilApache.compress;
import static de.wsi.e24.wifiholography.CaptureUtilApache.frequencyToIndex;
import static de.wsi.e24.wifiholography.CaptureUtilApache.rfft;
import static de.wsi.e24.wifiholography.CaptureUtilApache.score;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.complex.Complex;

import javafx.application.Platform;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;

/*
 * Low bandwidth (1083): 12900-13250 (350)
 * High bandwidth: 12300-13600 (1300)
 */

public class WavefrontAnalyzer {
	private SimpleObjectProperty<Integer> lowerLimitMHz, upperLimitMHz, frequencyGroups, frequencyOffsetMHz;
	private SimpleDoubleProperty referenceQuality;
	private SimpleObjectProperty<NormalizationMethod> referenceChannel;
	
	private volatile Thread mainThread;
	private volatile boolean stop;
	private volatile boolean error;
	
	
	public WavefrontAnalyzer() {
		lowerLimitMHz = new SimpleObjectProperty<>(170);
		upperLimitMHz = new SimpleObjectProperty<>(250);
		frequencyGroups = new SimpleObjectProperty<>(50);
		frequencyOffsetMHz =  new SimpleObjectProperty<>(5000);
		referenceQuality = new SimpleDoubleProperty(0.5);
		referenceChannel = new SimpleObjectProperty<>(NormalizationMethod.CHANNEL_1);
	}

	public void stop() {
		stop = true;
		if(mainThread != null) {
			mainThread.interrupt();
		}
	}
	
	public boolean fromRecording(Recording recording, Wavefront wavefront, DoubleProperty progress, boolean plotOnly) {
		stop = false;
		error = false;
		
		Platform.runLater(() -> progress.setValue(0));
		
		wavefront.processing().set(true);
		
		Settings settings = new Settings();
		wavefront.setFrequencies(settings.offset + settings.lower, settings.offset + settings.upper, settings.groups);
		
		int threadCount = Math.max(1, Math.min(Runtime.getRuntime().availableProcessors()-1, recording.getMaxReadingThreads()));
		ExecutorService service = Executors.newFixedThreadPool(threadCount);

		int size = recording.getCaptures().intValue();
		int indexGroup = 50;
		for(int i = 0; i < size; i+=indexGroup) {
			int index = i;
			int endIndex = Math.min(index+indexGroup, size-1);
			service.execute(() -> {
				try {
					analyze(recording, wavefront, index, endIndex, settings, plotOnly);
				} catch (Exception e) {
					if(error) return;
					error = true;
					Platform.runLater(() -> {
						Alert alert = new Alert(AlertType.ERROR);
						alert.setTitle("Error");
						alert.setTitle("Analysis failed");
						alert.setContentText("Some signals could not be analyzed.\n"+e);
						alert.show();
					});
				}
				Platform.runLater(() -> progress.setValue(index / (double) size));
			});
		}
		
		service.shutdown();
		mainThread = Thread.currentThread();
		try {
			service.awaitTermination(1000, TimeUnit.DAYS);
			return true;
		} catch (InterruptedException e) {
			if(stop) {
				service.shutdownNow();
			}
			else e.printStackTrace();
			return false;
		} finally {
			wavefront.processing().set(false);
		}
	}
	
	private class Settings
	{
		double lower = lowerLimitMHz.get() * 1e6;
		double upper = upperLimitMHz.get() * 1e6;
		int groups = frequencyGroups.get();
		double offset = frequencyOffsetMHz.get()*1e6;
		NormalizationMethod reference = referenceChannel.get();
		double minScore = referenceQuality.doubleValue();
	}
	
	


	private void analyze(Recording recording, Wavefront wavefront, int startIndex, int endIndex, Settings settings, boolean plotOnly) throws Exception
	{
		List<WavefrontPoint> points = new ArrayList<>();
		List<PointLocation> fails = new ArrayList<>();
		
		for(int i = startIndex; i <= endIndex; i++) {
			try {
				Capture capture = recording.getCapture(i);
				if(plotOnly) {
					capture.findLocation().ifPresent(loc -> fails.add(loc));
				}
				else analyzeSingle(capture, wavefront, settings, points, fails);
			} catch (IOException e) {
				System.err.println("Could not load capture "+i);
			}
		}
		
		synchronized(wavefront) {
			wavefront.getFailedCaptures().addAll(fails);
			wavefront.getPoints().addAll(points);
		}
	}
	
	private void analyzeSingle(Capture capture, Wavefront wavefront, Settings s, List<WavefrontPoint> points, List<PointLocation> fails) throws Exception {
		Optional<PointLocation> opLoc = capture.findLocation();
		if(opLoc.isPresent()) {
			PointLocation loc = opLoc.get();
			try {
				Complex[] signalFFT = rfft(capture.getScaledValuesD(s.reference.getSignalChannel()));
				int lowerLimit = frequencyToIndex(s.lower, signalFFT.length, capture.getSamplingRate());
				int upperLimit = frequencyToIndex(s.upper, signalFFT.length, capture.getSamplingRate());
				WavefrontPoint successPoint = null;
				Complex[] phases = new Complex[0];
				
				if(s.reference.isNormalized()) {
					Complex[] referenceFFT = rfft(capture.getScaledValuesD(s.reference.getNormalizationChannel()));
					float score = score(abs(referenceFFT), lowerLimit, upperLimit);
					if(score >= s.minScore) {
						// compress(fft2/fft1)[lowerLimit:upperLimit]
						phases = new Complex[upperLimit-lowerLimit];
						for(int i = lowerLimit; i < upperLimit; i++) {
							phases[i-lowerLimit] = signalFFT[i].divide(referenceFFT[i]);
						}
						phases = compress(phases, s.groups);
						successPoint = new WavefrontPoint(loc, score, phases);
					}
				}
				else {
					float score = score(abs(signalFFT), lowerLimit, upperLimit);
					if(score >= s.minScore) {
						// compress(fft2/fft1)[lowerLimit:upperLimit]
						phases = new Complex[upperLimit-lowerLimit];
						for(int i = lowerLimit; i < upperLimit; i++) {
							phases[i-lowerLimit] = signalFFT[i];
						}
						if(s.groups > 0) phases = compress(phases, s.groups);
						successPoint = new WavefrontPoint(loc, score, phases);
					}
				}
				
				if(successPoint != null) {
					points.add(successPoint);
				} else {
					fails.add(loc);
				}
				if(s.groups == 0 && wavefront.getFrequencies().length == 0) {
					wavefront.setFrequencies(s.offset+s.lower, s.offset+s.upper, phases.length);
				}
				
			} catch (Exception exc) {
				System.err.println("Point failed due to "+exc);
				exc.printStackTrace();
				fails.add(loc);
				throw exc;
			}
		}
	}
	

	public SimpleObjectProperty<Integer> getLowerLimit() {
		return lowerLimitMHz;
	}


	public SimpleObjectProperty<Integer> getUpperLimit() {
		return upperLimitMHz;
	}


	public SimpleDoubleProperty getReferenceQuality() {
		return referenceQuality;
	}


	public SimpleObjectProperty<Integer> getBinCount() {
		return frequencyGroups;
	}


	public SimpleObjectProperty<Integer> getFrequencyGroups() {
		return frequencyGroups;
	}


	public SimpleObjectProperty<Integer> getFrequencyOffsetMHz() {
		return frequencyOffsetMHz;
	}

	public SimpleObjectProperty<NormalizationMethod> getReferenceChannel() {
		return referenceChannel;
	}
	
	
}
