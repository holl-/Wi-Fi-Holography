package de.wsi.e24.wifiholography;

import java.io.IOException;
import java.util.Optional;

public interface Capture {

	Optional<PointLocation> findLocation();

	boolean existsChannel(int channelNumber);

	int getValueCount(int channelNumber) throws IOException;

	float[] getScaledValuesF(int channelNumber) throws IOException;

	double[] getScaledValuesD(int channelNumber) throws IOException;

	double getScale(int channelNumber);

	int getIndex();

	long getTime();

	double getWaitTime();

	double getSamplingRate();

	double[] getScale();

}