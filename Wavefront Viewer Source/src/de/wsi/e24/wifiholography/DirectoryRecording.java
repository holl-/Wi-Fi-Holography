package de.wsi.e24.wifiholography;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

public class DirectoryRecording extends Recording {
	private Path path; // directory containing this recording

	public DirectoryRecording(Path scanFile) throws IOException {
		path = scanFile.getParent();
		name = scanFile.getParent().getFileName().toString();
		load(Files.newInputStream(scanFile));
	}



	@Override
	protected Optional<PositionLog> loadPositionLog() {
		Path positionLogFile = path.resolve("positions.csv");
		if(Files.exists(positionLogFile)) {
			try {
				return Optional.of(new PositionLog(Files.newBufferedReader(positionLogFile)));
			} catch (IOException e) {
				e.printStackTrace();
			}
		} 
		return Optional.empty();
	}

	
	public Path getDataDirectory() {
		return path.resolve("data");
	}

	public Path getPath() {
		return path;
	}
	

	public InputStream readData(String relativePath) throws IOException {
		return Files.newInputStream(getDataDirectory().resolve(relativePath));
	}
	
	public boolean existsData(String relativePath) {
		return Files.exists(getDataDirectory().resolve(relativePath));
	}
	
	public long dataSize(String relativePath) throws IOException {
		return Files.size(getDataDirectory().resolve(relativePath));
	}
	
	public byte[] readAllDataBytes(String relativePath) throws IOException {
		return Files.readAllBytes(getDataDirectory().resolve(relativePath));
	}
	

	@Override
	public Path outputFile(String prefName) {
		return path.resolve(prefName);
	}



	@Override
	public void freeDetailMemory() {
		// This model does not cache any waveforms
	}



}
