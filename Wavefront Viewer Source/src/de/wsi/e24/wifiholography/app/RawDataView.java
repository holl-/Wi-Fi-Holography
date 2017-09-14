package de.wsi.e24.wifiholography.app;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import de.wsi.e24.wifiholography.Capture;
import de.wsi.e24.wifiholography.CaptureUtilJTransforms;
import de.wsi.e24.wifiholography.Recording;
import javafx.application.Platform;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.value.ObservableObjectValue;
import javafx.scene.Cursor;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.layout.StackPane;

public class RawDataView extends StackPane {
	private ObservableObjectValue<Recording> recording;
	private ObservableObjectValue<Integer> index;
	
	private SimpleDoubleProperty position;
	private SimpleBooleanProperty fft;

	private LineChart<Number, Number> captureGraph;
	private NumberAxis xAxis;
	
//	private CLHoloUtil clUtil;

	public RawDataView(ObservableObjectValue<Recording> recording, ObservableObjectValue<Integer> index) {
		this.recording = recording;
		this.index = index;
		
		position = new SimpleDoubleProperty(0.1);
		fft = new SimpleBooleanProperty(false);
		
		xAxis = new NumberAxis();
		xAxis.setForceZeroInRange(false);
        NumberAxis yAxis = new NumberAxis();
		
		captureGraph = new LineChart<>(xAxis, yAxis);
        captureGraph.setAnimated(false);
        captureGraph.setVerticalZeroLineVisible(false);
        captureGraph.setCreateSymbols(false);
        getChildren().add(captureGraph);

		index.addListener((v, o, n) -> updateGraph());
		recording.addListener((v, o, n) -> updateGraph());
		position.addListener((v, o, n) -> updateGraph());
		fft.addListener((v, o, n) -> {updateGraph(); updateAxis();});
		updateAxis();
	}

	private void updateGraph() {
		Cursor oldCursor = captureGraph.getScene().getCursor();
		Platform.runLater(() -> captureGraph.getScene().setCursor(Cursor.WAIT));
		
		new Thread(() -> {
			try {
				Recording rec = recording.get();
				int index = this.index.get();
				Capture capture;
				Platform.runLater(() -> captureGraph.getData().clear());
				try {
					capture = rec.getCapture(index);
				} catch (IOException e) {
					e.printStackTrace();
					captureGraph.setTitle("No capture with this index found");
					return;
				}
		
				captureGraph.setTitle(null);
		
				int channelNumber = 1;
				while (capture.existsChannel(channelNumber)) {
					XYChart.Series<Number, Number> series = new XYChart.Series<>();
					series.setName("Channel " + channelNumber);
					try {
						List<XYChart.Data<Number, Number>> points = getData(capture, channelNumber);
						series.getData().addAll(points);
					} catch (Exception e) {
						e.printStackTrace();
					}
					Platform.runLater(() -> captureGraph.getData().add(series));
					channelNumber++;
				}
			} catch(Throwable t) {
				System.gc();
				Platform.runLater(() -> {
					Alert alert = new Alert(AlertType.ERROR);
					alert.setContentText("Could not load signal.\n"+t);
					alert.setHeaderText("Loading failed.");
					alert.setTitle("Error");
					alert.show();
				});
				
			}

			Platform.runLater(() -> captureGraph.getScene().setCursor(oldCursor));
		}).start();
		
	}
	
	private List<XYChart.Data<Number, Number>> getData(Capture capture, int channelNumber) throws IOException {
		if(!fft.get()) {
			float[] values = capture.getScaledValuesF(channelNumber);
			return trim(values, 1000, 1.0 / capture.getSamplingRate() * 1e9);
		} 
		else {
			float[] values = CaptureUtilJTransforms.absRfft(capture.getScaledValuesD(channelNumber));
			return trim(values, 3000, CaptureUtilJTransforms.getFrequencySpacing(values.length, capture.getSamplingRate())/1e6);
		}
	}
	
	private List<XYChart.Data<Number, Number>> trim(float[] values, int visible, double xSpacing) {
		int offset, length;
		
		if(values.length <= visible) {
			offset = 0;
			length = values.length;
		} else {
			offset = (int) (position.get() * (values.length - visible));
			length = visible;
		}

		List<XYChart.Data<Number, Number>> points = IntStream.range(offset, offset+length)
				.mapToObj(i -> new XYChart.Data<Number, Number>(i * xSpacing, values[i]))
				.collect(Collectors.toList());
		
		return points;
	}

	public ObservableObjectValue<Recording> getRecording() {
		return recording;
	}

	public ObservableObjectValue<Integer> getIndex() {
		return index;
	}

	public SimpleDoubleProperty getPosition() {
		return position;
	}

	public SimpleBooleanProperty getFft() {
		return fft;
	}

	public LineChart<Number, Number> getCaptureGraph() {
		return captureGraph;
	}



	private void updateAxis() {
		xAxis.setLabel(fft.get() ? "Frequency / MHz" : "Time / ns");
	}
	
}
