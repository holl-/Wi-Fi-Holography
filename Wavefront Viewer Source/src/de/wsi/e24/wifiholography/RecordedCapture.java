package de.wsi.e24.wifiholography;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Properties;

public class RecordedCapture implements Capture {
	private Recording recording;
	private int index;

	private long time; // absolute time of this capture
	private double waitTime; // wait time in seconds
	private double samplingRate;
	private double[] scale;

	RecordedCapture(Recording recording, int index) throws IOException {
		this.recording = recording;
		this.index = index;
		load();
	}
	
	
	/* (non-Javadoc)
	 * @see de.wsi.e24.wifiholography.Capture#findLocation()
	 */
	@Override
	public Optional<PointLocation> findLocation() {
		if(recording.positionLog().isPresent()) {
			return recording.positionLog().get().getLocation(this);
		}
		else return Optional.empty();
	}

	/* (non-Javadoc)
	 * @see de.wsi.e24.wifiholography.Capture#existsChannel(int)
	 */
	@Override
	public boolean existsChannel(int channelNumber) {
		return recording.existsData(dataPath(channelNumber));
	}
	
	/* (non-Javadoc)
	 * @see de.wsi.e24.wifiholography.Capture#getValueCount(int)
	 */
	@Override
	public int getValueCount(int channelNumber) throws IOException {
		return (int) recording.dataSize(dataPath(channelNumber));
	}

	/**
	 * Not that the unsigned values are read as signed bytes.
	 * @param channelNumber
	 * @return
	 * @throws IOException
	 */
	private byte[] getRawData(int channelNumber) throws IOException {
		return recording.readAllDataBytes(dataPath(channelNumber));
	}

	// currently not used
	public int[] getUnscaledValues(int channelNumber) throws IOException {
		byte[] raw = getRawData(channelNumber);
		int[] result = new int[raw.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = unsignedByte(raw[i]);
		}
		return result;
	}
	
	/* (non-Javadoc)
	 * @see de.wsi.e24.wifiholography.Capture#getScaledValuesF(int)
	 */
	@Override
	public float[] getScaledValuesF(int channelNumber) throws IOException {
		byte[] raw = getRawData(channelNumber);
		float scale = (float) getScale(channelNumber);
		float[] result = new float[raw.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = unsignedByte(raw[i]) * scale;
		}
		return result;
	}
	
	/* (non-Javadoc)
	 * @see de.wsi.e24.wifiholography.Capture#getScaledValuesD(int)
	 */
	@Override
	public double[] getScaledValuesD(int channelNumber) throws IOException {
		byte[] raw = getRawData(channelNumber);
		double scale = getScale(channelNumber);
		double[] result = new double[raw.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = unsignedByte(raw[i]) * scale;
		}
		return result;
	}
	
	// currently not used
	public void getUnscaledValues(int channelNumber, FloatBuffer writeTo, int skip) throws IOException {
		byte[] raw = getRawData(channelNumber);
		for(int i = 0; i < raw.length; i++) {
			writeTo.put(i*(1+skip), unsignedByte(raw[i]));
		}
	}
	
	private static int unsignedByte(byte value) {
		if(value < 0) return (int)value + 256;
		else return value;
	}

	/* (non-Javadoc)
	 * @see de.wsi.e24.wifiholography.Capture#getScale(int)
	 */
	@Override
	public double getScale(int channelNumber) {
		return scale[channelNumber - 1];
	}
	
	Recording getRecording() {
		return recording;
	}

	/* (non-Javadoc)
	 * @see de.wsi.e24.wifiholography.Capture#getIndex()
	 */
	@Override
	public int getIndex() {
		return index;
	}

	/* (non-Javadoc)
	 * @see de.wsi.e24.wifiholography.Capture#getTime()
	 */
	@Override
	public long getTime() {
		return time;
	}

	/* (non-Javadoc)
	 * @see de.wsi.e24.wifiholography.Capture#getWaitTime()
	 */
	@Override
	public double getWaitTime() {
		return waitTime;
	}

	/* (non-Javadoc)
	 * @see de.wsi.e24.wifiholography.Capture#getSamplingRate()
	 */
	@Override
	public double getSamplingRate() {
		return samplingRate;
	}

	/* (non-Javadoc)
	 * @see de.wsi.e24.wifiholography.Capture#getScale()
	 */
	@Override
	public double[] getScale() {
		return scale;
	}

	private String dataPath(int channelNumber) {
		return index + "_ch" + channelNumber + ".dat";
	}

	private void load() throws IOException {
		Properties properties = new Properties();
		properties.load(recording.readData(index + "_info.txt"));
		time = (long) (1000 * Double.valueOf(properties.getProperty("time_abs")));
		waitTime = Double.valueOf(properties.getProperty("wait_time"));
		samplingRate = Double.valueOf(properties.getProperty("sampling_rate"));
		List<Double> scaleList = new ArrayList<>();
		int ch = 1;
		while (properties.containsKey("scale_channel_" + ch)) {
			scaleList.add(Double.valueOf(properties.getProperty("scale_channel_" + ch)));
			ch++;
		}
		scale = scaleList.stream().mapToDouble(d -> d).toArray();
	}
}
