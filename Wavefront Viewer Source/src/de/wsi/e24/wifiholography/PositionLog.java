package de.wsi.e24.wifiholography;

import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.StreamSupport;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;

public class PositionLog {
	private List<Entry> entries;
	
	public PositionLog(Reader fileReader) throws IOException {
		entries = new ArrayList<>();
		load(fileReader);
	}
	
	public Optional<PointLocation> getLocation(RecordedCapture capture) {
		Recording rec = capture.getRecording();
		long time = capture.getTime();
		double x = -1, y = -1;
		
		if(time <= getStartTime()) {
			x = entries.get(0).getX();
			y = entries.get(0).getY();
		} else if(time >= getEndTime()) {
			x = entries.get(entries.size()-1).getX();
			y = entries.get(entries.size()-1).getY();
		} else {
			int i = findIndex(time);
			Entry prev = entries.get(i-1);
			Entry next = entries.get(i);
			double factor = (time - prev.getTime()) / (double)(next.getTime()-prev.getTime());
			x = (1-factor) * prev.getX() + factor * next.getX();
			y = (1-factor) * prev.getY() + factor * next.getY();
		}
		if(x < 0 || y < 0 || x > rec.getWidth() || y > rec.getHeight()) {
			System.out.println("Point out of bounds: "+capture.getIndex());
			return Optional.empty();
		}
		else return Optional.of(new PointLocation(capture, time, x, y));
	}
	
	private int findIndex(long time) {
		int stepSize = 10000;
		int max = 0;
		int size = entries.size();
		while(stepSize > 0) {
			if(max+stepSize < size && entries.get(max+stepSize).getTime() < time) {
				max += stepSize;
			}
			else {
				stepSize /= 10;
			}
		}
		return max;
	}
	
	public long getStartTime() {
		return entries.get(0).getTime();
	}
	
	public long getEndTime() {
		return entries.get(entries.size()-1).getTime();
	}
	
	private void load(Reader reader) throws IOException {
		CSVParser parser = new CSVParser(reader, CSVFormat.DEFAULT);
		StreamSupport.stream(parser.spliterator(), false).skip(1).forEach(line -> {
			try {
				long time = (long) (1000*Double.valueOf(line.get(0)));
				double x = Double.valueOf(line.get(1));
				double y = Double.valueOf(line.get(2));
				entries.add(new Entry(time, x, y));
			} catch(Exception exc) {
				System.err.println("The position log file may not be complete. Some data was not read.");
//				exc.printStackTrace();
			}
		});
		parser.close();
	}

	private class Entry {
		private long time;
		private double x, y;

		public Entry(long time, double x, double y) {
			this.time = time;
			this.x = x;
			this.y = y;
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
}
