package de.wsi.e24.wifiholography;

public enum NormalizationMethod {
	CHANNEL_1("Normalize by channel 1", 2, 1),
	CHANNEL_2("Normalize by channel 2", 1, 2), // for
																							// recording
	POLARIZATION_X("Unnormalized channel 1 (X)", 1, -1),
	POLARIZATION_Y("Unnormalized channel 2 (Y)", 2, -1),
	POLARIZATION_Z("Unnormalized channel 3 (Z)", 3, -1); // for
																										// simulation

	private String name;
	private int signalChannel, normalizationChannel;

	private NormalizationMethod(String name, int signalChannel, int normalizationChannel) {
		this.name = name;
		this.signalChannel = signalChannel;
		this.normalizationChannel = normalizationChannel;
	}

	public String getName() {
		return name;
	}

	@Override
	public String toString() {
		return name;
	}

	public boolean isNormalized() {
		return normalizationChannel >= 0;
	}

	public int getSignalChannel() {
		return signalChannel;
	}

	public int getNormalizationChannel() {
		return normalizationChannel;
	}

}
