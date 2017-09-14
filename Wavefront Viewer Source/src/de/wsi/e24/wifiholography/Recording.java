package de.wsi.e24.wifiholography;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Optional;
import java.util.Properties;

import javafx.application.Platform;
import javafx.beans.property.SimpleIntegerProperty;

public abstract class Recording {
	protected String name;
	protected double width, height; // in meters
	protected double ySpeed;
	protected int steps; // number of horizontal steps
	protected long time; // start time of recording
	protected SimpleIntegerProperty captures = new SimpleIntegerProperty(0);
	private Optional<PositionLog> positionLog;


	

	public void countFiles() {
		int stepCount = 1000;
		int max = 0;
		
		while(stepCount > 0) {
			if(existsData((max+stepCount)+"_info.txt")) {
				max += stepCount;
			} else {
				stepCount /= 10;
			}
		}
		int maxF = max;
		Platform.runLater(() -> captures.set(maxF));
	}

	protected void load(InputStream in) throws IOException {
		Properties properties = new Properties();
		properties.load(in);
		width = Double.valueOf(properties.getProperty("width", "-1"));
		height = Double.valueOf(properties.getProperty("height", "-1"));
		ySpeed = Double.valueOf(properties.getProperty("y_speed", "-1"));
		steps = Integer.valueOf(properties.getProperty("steps", "-1"));
		time = (long)(1000*Double.valueOf(properties.getProperty("time", "-1")));
	}
	
	public abstract void freeDetailMemory();
	
	public abstract InputStream readData(String relativePath) throws IOException;
	
	public abstract boolean existsData(String relativePath);
	
	public abstract long dataSize(String relativePath) throws IOException;
	
	public byte[] readAllDataBytes(String relativePath) throws IOException {
		int size = (int) dataSize(relativePath);
		byte[] bytes = new byte[size];
		InputStream in = readData(relativePath);
		int read = 0;
		int len;
		while((len = in.read(bytes, read, size-read)) != -1 && read < size) {
			read += len;
		}
		return bytes;
	}
	
	public abstract Path outputFile(String prefName);
	
	public Optional<PositionLog> positionLog() {
		if(positionLog == null) {
			positionLog = loadPositionLog();
		}
		return positionLog;
	}
	
	protected abstract Optional<PositionLog> loadPositionLog();
	
	public Capture getCapture(int index) throws IOException {
		return new RecordedCapture(this, index);
	}

	public String getName() {
		return name;
	}

	public double getWidth() {
		return width;
	}

	public double getHeight() {
		return height;
	}

	public double getySpeed() {
		return ySpeed;
	}

	public int getSteps() {
		return steps;
	}

	public long getTime() {
		return time;
	}

	public SimpleIntegerProperty getCaptures() {
		return captures;
	}

	public int getDefaultFrequencyOffset() {
		return 5000;
	}

	public NormalizationMethod getDefaultChannel() {
		return NormalizationMethod.CHANNEL_1;
	}

	public int getDefaultLowerFrequency() {
		return 170;
	}

	public Integer getDefaultUpperFrequency() {
		return 250;
	}

	public int getDefaultFrequencyBins() {
		return 50;
	}

	public int getMaxReadingThreads() {
		return 3;
	}

}
