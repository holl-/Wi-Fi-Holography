package de.wsi.e24.wifiholography;

import org.apache.commons.math3.complex.Complex;

public class WavefrontPoint {
	private PointLocation location;
	// private double meanFrequency;
	private double correlationQuality;

	private Complex[] phases; // TODO other polarization

	public WavefrontPoint(PointLocation location, double correlationQuality, Complex[] phases) {
		this.location = location;
		this.correlationQuality = correlationQuality;
		this.phases = phases;
	}

	public PointLocation getLocation() {
		return location;
	}

	public double getCorrelationQuality() {
		return correlationQuality;
	}

	public Complex[] getPhases() {
		return phases;
	}

}
