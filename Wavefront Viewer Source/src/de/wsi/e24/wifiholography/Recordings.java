package de.wsi.e24.wifiholography;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;

public class Recordings {

	
	public static List<Recording> getRecordings(Path dir) throws IOException {
		return Files.list(dir).map(path -> {
			try {
				if(Files.isDirectory(path)) {
					Path scanFile = path.resolve("scan.txt");
					if(Files.exists(scanFile)) {
						return new DirectoryRecording(scanFile);
					}
					else return null;
				}
				else if(path.toString().toLowerCase().endsWith(".rec.zip")) {
					return new ZipRecording(path);
				}
				else if(path.toString().toLowerCase().endsWith(".mat")) {
					return new SimulationResult(path);
				}
				else return null;
			} catch(IOException e) {
				e.printStackTrace();
				return null;
			}
		})
				.filter(rec -> rec != null).collect(Collectors.toList());
	}
	
}
