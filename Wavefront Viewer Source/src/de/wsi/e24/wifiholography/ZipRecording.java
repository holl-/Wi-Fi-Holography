package de.wsi.e24.wifiholography;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.Optional;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;

public class ZipRecording extends Recording {
	private Path file;
	private ZipFile zip;
	private String prefix = "";
	
	public ZipRecording(Path file) throws ZipException, IOException {
		this.file = file;
		zip = new ZipFile(file.toFile());
		name = file.getFileName().toString();
		if(name.toLowerCase().endsWith(".rec.zip")) {
			name = name.substring(0, name.length()-".rec.zip".length());
		}
		
		ZipEntry scanFile = zip.getEntry("scan.txt");
		if(scanFile == null) {
			ZipEntry randomEntry = zip.stream().findAny().orElse(null);
			if(randomEntry == null) throw new IllegalArgumentException("Zip file is empty");
			prefix = randomEntry.getName();
			prefix = prefix.substring(0, prefix.indexOf("/")+1);
			scanFile = zip.getEntry(prefix+"scan.txt");
		}
		if(scanFile == null) {
			throw new IllegalArgumentException("No scan file in "+file);
		}
		load(zip.getInputStream(scanFile));
	}


	@Override
	protected Optional<PositionLog> loadPositionLog() {
		ZipEntry positionLogFile = zip.getEntry(prefix+"positions.csv");
		if(positionLogFile != null) {
			try {
				return Optional.of(new PositionLog(new BufferedReader(new InputStreamReader(zip.getInputStream(positionLogFile)))));
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return Optional.empty();
	}
	
	
	public Path getFile() {
		return file;
	}
	
	private ZipEntry dataEntry(String relativePath) {
		return zip.getEntry(prefix+"data/"+relativePath);
	}
	

	@Override
	public InputStream readData(String relativePath) throws IOException {
		return zip.getInputStream(dataEntry(relativePath));
	}

	@Override
	public boolean existsData(String relativePath) {
		return dataEntry(relativePath) != null;
	}

	@Override
	public long dataSize(String relativePath) throws IOException {
		return dataEntry(relativePath).getSize();
	}

	@Override
	public Path outputFile(String prefName) {
		return file.resolveSibling(name+"_"+prefName);
	}


	@Override
	public void freeDetailMemory() {
		// the zip file is not cached in memory.
	}

}
