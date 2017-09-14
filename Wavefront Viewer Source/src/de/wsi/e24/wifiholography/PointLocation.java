package de.wsi.e24.wifiholography;

public class PointLocation {
	private Capture capture;
	private long time;
	private double x, y;

	public PointLocation(Capture capture, long time, double x, double y) {
		this.capture = capture;
		this.time = time;
		this.x = x;
		this.y = y;
	}

	public Capture getCapture() {
		return capture;
	}
	
	public int getCaptureIndex() {
		return capture.getIndex();
	}

	public long getTime() {
		return time;
	}

	public double getX() {
		return x;
	}

	public double getY() {
		return y;
	}

}
