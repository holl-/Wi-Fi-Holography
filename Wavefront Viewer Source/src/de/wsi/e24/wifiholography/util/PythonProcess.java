package de.wsi.e24.wifiholography.util;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;

public class PythonProcess {
	private SimpleStringProperty output = new SimpleStringProperty("");
	private Path pythonFile;
	
	public PythonProcess(Path pythonFile) {
		this.pythonFile = pythonFile;
	}
	
	public void run(List<String> args) throws IOException, InterruptedException {
		List<String> commands = new ArrayList<>();
		commands.add("python");
		commands.add(pythonFile.toAbsolutePath().toString());
		commands.addAll(args);
		ProcessBuilder pb = new ProcessBuilder(commands);
		pb.redirectErrorStream(true);
		pb.directory(new File("").getAbsoluteFile());
		Process process = pb.start();
		
		BufferedReader processOut = new BufferedReader(new InputStreamReader(process.getInputStream()));
		new Thread(() -> writeOut(processOut)).start();
		
	}

	private void writeOut(BufferedReader processOut) {
		String line;
		try {
			while ((line = processOut.readLine()) != null) {
				String oldText = output.get();
				String newText = oldText + line + "\n";
				output.set(newText);
			}
			System.out.println("Stream closed");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public StringProperty output() {
		return output;
	}
}
