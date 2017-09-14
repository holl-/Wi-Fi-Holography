package de.wsi.e24.wifiholography;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.stream.IntStream;

import org.apache.commons.math3.complex.Complex;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;

public class Wavefront {
	private Recording recording;
	private ObservableList<WavefrontPoint> points;
	private ObservableList<PointLocation> failedCaptures;
	private double[] frequencies;
	private BooleanProperty processing;

	public Wavefront(Recording recording) {
		this.recording = recording;
		points = FXCollections.observableArrayList();
		failedCaptures = FXCollections.observableArrayList();
		processing = new SimpleBooleanProperty();
	}

	public ObservableList<WavefrontPoint> getPoints() {
		return points;
	}

	public ObservableList<PointLocation> getFailedCaptures() {
		return failedCaptures;
	}

	public Recording getRecording() {
		return recording;
	}
	
	public BooleanProperty processing() {
		return processing;
	}
	

	public double[] getFrequencies() {
		return frequencies;
	}

	public void setFrequencies(double[] frequencies) {
		this.frequencies = frequencies;
	}

	public void setFrequencies(double lowerFreq, double upperFreq, int groups) {
		double range = upperFreq - lowerFreq;
		frequencies = IntStream.range(0, groups).mapToDouble(i -> {
			return lowerFreq + range * (i+0.5) / groups;
		}).toArray();
	}
	
	public void write() throws IOException {
		Path path = getDefaultOutputFile(recording);
		write(Files.newBufferedWriter(path));
	}
	
	private Path getDefaultOutputFile(Recording recording) {
		return recording.outputFile("wavefront_"+frequencies.length+".csv");
	}

	public void write(Writer out) throws IOException {
		System.out.println("Writing wavefront");
		BufferedWriter writer = new BufferedWriter(out);
		writer.write("Index,X,Y");
		for(int i = 0; i < frequencies.length; i++) {
			writer.write(",Re(frequency "+i+"),Im(frequency "+i+")");
		}
		writer.newLine();
		writer.write("0,0,0");
		for(double frequency : frequencies) {
			writer.write(","+frequency+","+frequency);
		}
		
		writer.newLine();
		
		for(WavefrontPoint point : new ArrayList<>(points)) {
			writer.write(point.getLocation().getCaptureIndex()+","+point.getLocation().getX()+","+point.getLocation().getY());
			for(Complex phase : point.getPhases()) {
				writer.write(","+phase.getReal()+","+phase.getImaginary());
			}
			writer.newLine();
		}
		writer.close();
	}
}
