package de.wsi.e24.wifiholography;

import java.io.IOException;
import java.util.Optional;

public class SimulationResultPoint implements Capture {
	private SimulationResult simulation;
	private int x, y;

	public SimulationResultPoint(SimulationResult simulation, int x, int y) {
		this.simulation = simulation;
		this.x = x;
		this.y = y;
	}

	@Override
	public Optional<PointLocation> findLocation() {
		return Optional.of(new PointLocation(this, getTime(), simulation.positionsX[x], simulation.positionsY[y]));
	}

	@Override
	public boolean existsChannel(int channelNumber) {
		return channelNumber > 0 && channelNumber <= 3;
	}

	@Override
	public int getValueCount(int channelNumber) throws IOException {
		return simulation.getTimeCount();
	}

	@Override
	public float[] getScaledValuesF(int channelNumber) throws IOException {
		return simulation.getValuesF(x, y, channelNumber-1);
	}

	@Override
	public double[] getScaledValuesD(int channelNumber) throws IOException {
		return simulation.getValuesD(x, y, channelNumber-1);
	}

	@Override
	public double getScale(int channelNumber) {
		return 1;
	}

	@Override
	public int getIndex() {
		return x + y * simulation.getPointsX();
	}

	@Override
	public long getTime() {
		return simulation.getTime();
	}

	@Override
	public double getWaitTime() {
		return 0;
	}

	@Override
	public double getSamplingRate() {
		return simulation.getSamplingRate();
	}

	@Override
	public double[] getScale() {
		return new double[] { 1.0 };
	}

}
