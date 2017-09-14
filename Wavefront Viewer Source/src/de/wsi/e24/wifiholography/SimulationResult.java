package de.wsi.e24.wifiholography;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5SimpleReader;

public class SimulationResult extends Recording {
	private Path file;
	float[] positionsX, positionsY, times;
	private double samplingRate;
	private IHDF5SimpleReader reader;
	private List<float[]> fields; // Ex, Ey, Ez (values vs t, z, y, x)
	
	
	public SimulationResult(Path file) {
		this.file = file;
		name = file.getFileName().toString();
		if(name.toLowerCase().endsWith(".mat")) {
			name = name.substring(0, name.length()-".mat".length());
		}
		
		reader = HDF5Factory.openForReading(file.toFile());
		float[] valuesX = reader.readFloatArray("x");
		float[] valuesY = reader.readFloatArray("y");
		float[] valuesZ = reader.readFloatArray("z");
		
		if(valuesZ.length <= 1) {
			positionsX = valuesX;
			positionsY = valuesY;
		} else {
			positionsY = valuesZ;
			positionsX = (valuesY.length <= 1) ? valuesX : valuesY;
		}
		normalize(positionsX);
		normalize(positionsY);
		
		times = reader.readFloatArray("t");
		
		width = positionsX[positionsX.length-1] - positionsX[0];
		height = positionsY[positionsY.length-1] - positionsY[0];
		samplingRate = times.length / (times[times.length-1] - times[0]);
		ySpeed = positionsY.length;
		steps = positionsX.length;
		try {
			time = Files.getLastModifiedTime(file).toMillis();
		} catch (IOException e) {
			time = -1;
		}
		captures.set(getPointsX() * getPointsY());
	}
	
	private void normalize(float[] positions) {
		float min = positions[0];
		for(int i = 0; i < positions.length; i++) {
			positions[i] -= min;
		}
	}

	public Capture getCapture(int index) throws IOException {
		int x = index % getPointsX();
		int y = index / getPointsX();
		return new SimulationResultPoint(this, x, y);
	}
	
	public float[] getValuesF(int x, int y, int fieldIndex) {
		if(fields == null) {
			loadField();
		}
		
		float[] field = fields.get(fieldIndex);
		float[] result = new float[times.length];
		int w = positionsX.length;
		int h = positionsY.length;
		for(int i = 0; i < result.length; i++) {
			result[i] = field[i*w*h + y*w + x] / 1e6f;
		}
		return result;
	}

	public double[] getValuesD(int x, int y, int fieldIndex) {
		if(fields == null) {
			loadField();
		}
		
		float[] field = fields.get(fieldIndex);
		double[] result = new double[times.length];
		int w = positionsX.length;
		int h = positionsY.length;
		for(int i = 0; i < result.length; i++) {
			result[i] = field[i*w*h + y*w + x] / 1e6;
		}
		return result;
	}
	
	private void loadField() {
		fields = new ArrayList<>(3);
		fields.add(reader.readFloatArray("Ex"));
		fields.add(reader.readFloatArray("Ey"));
		fields.add(reader.readFloatArray("Ez"));
	}

	@Override
	public void countFiles() {
		return; // no need to count
	}
	
	@Override
	public InputStream readData(String relativePath) throws IOException {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean existsData(String relativePath) {
		throw new UnsupportedOperationException();
	}

	@Override
	public long dataSize(String relativePath) throws IOException {
		throw new UnsupportedOperationException();
	}

	@Override
	public Path outputFile(String prefName) {
		return file.resolveSibling(name+"_"+prefName);
	}

	@Override
	protected Optional<PositionLog> loadPositionLog() {
		return Optional.empty();
	}
	
	public Path getFile() {
		return file;
	}
	
	public double getSamplingRate() {
		return samplingRate;
	}
	
	public int getPointsX() {
		return positionsX.length;
	}
	
	public int getPointsY() {
		return positionsY.length;
	}
	
	public int getTimeCount() {
		return times.length;
	}

	@Override
	public int getDefaultFrequencyOffset() {
		return 0;
	}
	@Override
	public NormalizationMethod getDefaultChannel() {
		return NormalizationMethod.POLARIZATION_Z;
	}

	@Override
	public int getDefaultLowerFrequency() {
		if(samplingRate > 10e9) {
			return 5000-40;
		}
		else return 2472-11;
	}

	@Override
	public Integer getDefaultUpperFrequency() {
		if(samplingRate > 10e9) {
			return 5000+40;
		}
		else return 2472+11;
	}

	@Override
	public int getDefaultFrequencyBins() {
		return 0;
	}
	
	@Override
	public int getMaxReadingThreads() {
		return Integer.MAX_VALUE;
	}

	@Override
	public void freeDetailMemory() {
		fields = null;
	}
}
