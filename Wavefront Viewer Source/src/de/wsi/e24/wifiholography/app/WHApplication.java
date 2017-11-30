package de.wsi.e24.wifiholography.app;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.ResourceBundle;
import java.util.stream.Collectors;

import de.wsi.e24.wifiholography.NormalizationMethod;
import de.wsi.e24.wifiholography.Recording;
import de.wsi.e24.wifiholography.Recordings;
import de.wsi.e24.wifiholography.Wavefront;
import de.wsi.e24.wifiholography.WavefrontAnalyzer;
import de.wsi.e24.wifiholography.util.PythonProcess;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.beans.Observable;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory.IntegerSpinnerValueFactory;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;
import javafx.util.converter.DateTimeStringConverter;
import javafx.util.converter.NumberStringConverter;

public class WHApplication extends Application implements Initializable, RequestHandler{
	@FXML private TabPane tabs;
	@FXML private Tab tabRaw, tabWavefront;
	
	@FXML private TableView<Recording> recordingsTable;
	@FXML private TableColumn<Recording, String> nameCol;
	@FXML private TableColumn<Recording, String> sizeCol;
	@FXML private TableColumn<Recording, String> yspeedCol;
	@FXML private TableColumn<Recording, String> stepsCol;
	@FXML private TableColumn<Recording, String> timeCol;
	@FXML private TableColumn<Recording, String> capturesCol;
	
	@FXML private Spinner<Integer> captureSelect;
	private IntegerSpinnerValueFactory selectedCapture;
	@FXML private Slider capturePos;
	@FXML private CheckBox captureFFT;
	@FXML private StackPane captureGraphContainer;
	private RawDataView rawDataView;
	
	@FXML private Spinner<Integer> lowerFLimit, upperFLimit, frequencyGroups, frequencyOffset;
	@FXML private Slider referenceQuality;
	@FXML private ProgressBar correlationProgressBar;
	@FXML private ComboBox<NormalizationMethod> referenceChannel;
	private WavefrontAnalyzer analyzer;
	
	@FXML private Slider radiusSlider;
	@FXML private StackPane wavefrontContainer;
	private WavefrontView wavefrontView;
	private SimpleObjectProperty<Wavefront> activeWavefront;
	
	@FXML private ListView<String> processedFileList;
	@FXML private TextField pythonFileArg, pythonArguments;
	@FXML private CheckBox pythonMultiThreaded;
	@FXML private TextArea pythonOutput;
	
	
	@Override
	public void start(Stage primaryStage) throws Exception {
		FXMLLoader loader = new FXMLLoader(getClass().getResource("Application.fxml"));
		loader.setController(this);
		
		Scene scene = new Scene(loader.load(), 800, 600);
		primaryStage.setScene(scene);
		primaryStage.setOnCloseRequest(e -> System.exit(0));
		primaryStage.setTitle("Wavefront Viewer");
		primaryStage.show();
	}

	public static void main(String[] args) {
		launch(args);
	}

	@Override
	public void initialize(URL location, ResourceBundle resources) {
		DateTimeStringConverter dates = new DateTimeStringConverter();
		DecimalFormat df = new DecimalFormat("#.00");
		
		nameCol.setCellValueFactory(rec -> new SimpleStringProperty(rec.getValue().getName()));
		sizeCol.setCellValueFactory(rec -> new SimpleStringProperty(df.format(rec.getValue().getWidth())+" x "+df.format(rec.getValue().getHeight())));
		yspeedCol.setCellValueFactory(rec -> new SimpleStringProperty(rec.getValue().getySpeed()+""));
		stepsCol.setCellValueFactory(rec -> new SimpleStringProperty(rec.getValue().getSteps()+""));
		timeCol.setCellValueFactory(rec -> new SimpleStringProperty(dates.toString(new Date(rec.getValue().getTime()))));
		capturesCol.setCellValueFactory(rec -> {
			SimpleStringProperty p = new SimpleStringProperty();
			p.bindBidirectional(rec.getValue().getCaptures(), new NumberStringConverter());
			return p;
			});
		recordingsTable.getSelectionModel().selectedItemProperty().addListener((p, o, n) -> {
			// when a recording is selected
			
			// set new parameters
			selectedCapture.maxProperty().bind(n.getCaptures());
			frequencyOffset.getValueFactory().setValue(n.getDefaultFrequencyOffset());
			referenceChannel.setValue(n.getDefaultChannel());
			lowerFLimit.getValueFactory().setValue(n.getDefaultLowerFrequency());
			upperFLimit.getValueFactory().setValue(n.getDefaultUpperFrequency());
			frequencyGroups.getValueFactory().setValue(n.getDefaultFrequencyBins());
			
			// update file list
			updateProcessedFileList();
			
			// free old memory
			if(o != null) {
				o.freeDetailMemory();
			}
		});
//		capturesCol.setCellValueFactory(new PropertyValueFactory<Recording, IntegerProperty>("captures"));
//		Callback<TableColumn<Object, Integer>, TableCell<Object, Integer>> conv = TextFieldTableCell.forTableColumn(new IntegerStringConverter());
//		capturesCol.setCellFactory();
		
		captureSelect.setValueFactory(selectedCapture = new IntegerSpinnerValueFactory(0, 0));
        rawDataView = new RawDataView(recordingsTable.getSelectionModel().selectedItemProperty(), selectedCapture.valueProperty());
        capturePos.valueProperty().bindBidirectional(rawDataView.getPosition());
        captureFFT.selectedProperty().bindBidirectional(rawDataView.getFft());
        captureGraphContainer.getChildren().add(rawDataView);

        analyzer = new WavefrontAnalyzer();
        bind(lowerFLimit, analyzer.getLowerLimit());
        bind(upperFLimit, analyzer.getUpperLimit());
        bind(frequencyGroups, analyzer.getFrequencyGroups());
        bind(frequencyOffset, analyzer.getFrequencyOffsetMHz());
        referenceQuality.valueProperty().bindBidirectional(analyzer.getReferenceQuality());
        referenceChannel.setItems(FXCollections.observableArrayList(NormalizationMethod.values()));
        System.setProperty("glass.accessible.force", "false");
        referenceChannel.valueProperty().bindBidirectional(analyzer.getReferenceChannel());
        
        
        wavefrontView = new WavefrontView(this, activeWavefront = new SimpleObjectProperty<>(), referenceQuality.valueProperty());
        radiusSlider.valueProperty().bindBidirectional(wavefrontView.getRadius());
        wavefrontContainer.getChildren().add(wavefrontView);
        
        processedFileList.getSelectionModel().selectedItemProperty().addListener((p, o, n) -> {
        	pythonFileArg.setText(n);
        });
		
        recordingsTable.setPlaceholder(new Label("Loading scan files..."));
		new Thread(() -> discoverRecordings()).start();
	}
	
	private void updateProcessedFileList() {
		processedFileList.getItems().clear();
		Recording rec = recordingsTable.getSelectionModel().getSelectedItem();
		try {
			processedFileList.getItems().addAll(Files.list(rec.outputFile("test").getParent()).filter(path -> {
					String name = path.getFileName().toString().toLowerCase();
					System.out.println(name);
					return name.startsWith(rec.getName().toLowerCase()) && name.endsWith(".csv");
				}).map(path -> path.getFileName().toString()).collect(Collectors.toList()));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private void discoverRecordings() {
		ObservableList<Recording> recordings = FXCollections.observableArrayList(rec -> new Observable[]{rec.getCaptures()});
		Path path = isJar() ? Paths.get("").toAbsolutePath() : Paths.get("E:/Recordings/Software Test").toAbsolutePath();
		try {
			recordings.addAll(Recordings.getRecordings(path));
		} catch (IOException e) {
			e.printStackTrace();
		}
		recordingsTable.setItems(recordings);
        recordingsTable.setPlaceholder(new Label("No scans found."));
        new Thread(() -> recordings.parallelStream().forEach(rec -> rec.countFiles())).start();
	}
	
	
	private void bind(Spinner<Integer> spinner, ObjectProperty<Integer> property) {
        IntegerSpinnerValueFactory factory = new IntegerSpinnerValueFactory(0, 1000000);
        spinner.setValueFactory(factory);
        factory.valueProperty().bindBidirectional(property);
	}

	private boolean isJar() {
		return getClass().getResource("Application.fxml").getProtocol().toLowerCase().equals("jar");
	}

	@FXML
	public void layoutWavefront() {
		createWavefront(true);
	}
	
	@FXML
	public void createWavefront() {
		createWavefront(false);
	}

	private void createWavefront(boolean plotOnly) {
		Recording recording = recordingsTable.getSelectionModel().getSelectedItem();
		Wavefront wavefront = new Wavefront(recording);
		activeWavefront.set(wavefront);
		
		new Thread(() -> {
			boolean success = analyzer.fromRecording(recording, wavefront, correlationProgressBar.progressProperty(), plotOnly);
			if(!plotOnly && success) {
				
				Platform.runLater(() -> {
					try {
						wavefront.write();
						System.out.println("Wavefront written to file");
						updateProcessedFileList();
					} catch (Exception e) {
						e.printStackTrace();
					}
				});
			}
			Platform.runLater(() -> tabs.getSelectionModel().select(tabWavefront));
			}).start();
	}
	

	@Override
	public void showRaw(int index) {
		selectedCapture.setValue(index);
		tabs.getSelectionModel().select(tabRaw);
	}
	
	@FXML
	public void stopCreation() {
		analyzer.stop();
	}
	
	@FXML
	public void quit() {
		System.exit(0);
	}
	
	@FXML
	public void about() {
		Alert dialog = new Alert(AlertType.INFORMATION);
		dialog.setTitle("About Wavefront Viewer");
		dialog.setHeaderText("Wavefront Viewer");
		dialog.setContentText("Author: Philipp Holl, E24 TU München\n"
				+ "\nThis software was written to preprocess, inspect and convert raw recording data and simulation results. "
				+ "Back-propagation is implemented in Python."
				+ "\n\nPut executable JAR next to wavefront files.\n"
				+ "The following input files are supported."
				+ "\n- Recording: file structure and .rec.zip"
				+ "\n- Simulation: .mat"
				+ "\n\nThis project is available on GitHub."
				+ "\nhttps://github.com/holl-/Wi-Fi-Holography");
		dialog.show();
	}
	
	@FXML
	public void launchPropagation() {
		String csvFile;
		if(pythonFileArg.getText().trim().isEmpty()) {
			if(processedFileList.getItems().isEmpty()) return;
			String firstFile = processedFileList.getItems().get(0);
			Recording recording = recordingsTable.getSelectionModel().getSelectedItem();
			if(recording == null) return;
			csvFile = recording.outputFile(firstFile).toAbsolutePath().toString();
		}
		else {
			csvFile = Paths.get(pythonFileArg.getText()).toAbsolutePath().toString();
		}
		Path pythonFile = Paths.get("Propagation/propagation_launcher.py").toAbsolutePath();
		
		// Build command line
		List<String> arguments = new ArrayList<>();
		arguments.add(csvFile);
		if(pythonMultiThreaded.isSelected()) {
			arguments.add("--multithreaded");
		}
		arguments.addAll(Arrays.asList(pythonArguments.getText().split(" ")));
		
		PythonProcess process = new PythonProcess(pythonFile);
		pythonOutput.textProperty().unbind();
		pythonOutput.textProperty().bind(process.output());
		try {
			process.run(arguments);
		} catch (IOException | InterruptedException e) {
			Alert dialog = new Alert(AlertType.ERROR);
			dialog.setTitle("Error");
			dialog.setHeaderText("Error launching Python");
			dialog.setContentText("Python could not be started. "+e+""
					+ "\nSee command line for details.");
			dialog.show();
			e.printStackTrace();
		}
	}
	
}
