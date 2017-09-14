package de.wsi.e24.wifiholography.app;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import de.wsi.e24.wifiholography.PointLocation;
import de.wsi.e24.wifiholography.Wavefront;
import de.wsi.e24.wifiholography.WavefrontPoint;
import javafx.application.Platform;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.value.ObservableDoubleValue;
import javafx.beans.value.ObservableObjectValue;
import javafx.collections.ListChangeListener;
import javafx.scene.Cursor;
import javafx.scene.Node;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.XYChart.Data;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;
import javafx.scene.shape.Circle;

public class WavefrontView extends StackPane {
	private ObservableObjectValue<Wavefront> wavefront;
	private SimpleDoubleProperty radius;

	private ScatterChart<Number, Number> scatter;
	private ArrayList<Data<Number, Number>> offlineList;
	private NumberAxis xAxis, yAxis;
	
	private ObservableDoubleValue minQuality;
	private Map<Node, String> tooltipMap = new HashMap<>();
	private Tooltip tooltip = new Tooltip();

	private RequestHandler handler;

	public WavefrontView(RequestHandler handler, ObservableObjectValue<Wavefront> wavefront, ObservableDoubleValue minQuality) {
		this.handler = handler;
		this.wavefront = wavefront;
		this.minQuality = minQuality;

		radius = new SimpleDoubleProperty(5);

		xAxis = new NumberAxis();
		yAxis = new NumberAxis();
		xAxis.setLabel("X / m");
		yAxis.setLabel("Y / m");

		scatter = new ScatterChart<>(xAxis, yAxis);
		scatter.setOnMouseMoved(e -> {
			Node node = e.getPickResult().getIntersectedNode();
			String text = tooltipMap.get(node);
			if(text != null) {
				tooltip.setText(text);
				tooltip.show(node, e.getScreenX() + 20, e.getScreenY());
			} else {
				tooltip.hide();
			}
		});
		getChildren().add(scatter);
//		scatter.setAnimated(false);

		wavefront.addListener((w, o, n) -> updateWavefront());
	}

	private void updateWavefront() {
		scatter.getData().clear();

		if (wavefront.get() == null)
			return;

		Wavefront w = wavefront.get();
		xAxis.setUpperBound(w.getRecording().getWidth());
		xAxis.setAutoRanging(false);

		// Not included
		XYChart.Series<Number, Number> fails = new XYChart.Series<>();
		fails.setName("Not included");
		fails.getData().addAll(w.getFailedCaptures().stream().map(loc -> createData(loc)).collect(Collectors.toList()));
		scatter.getData().add(fails);

		// Included
		XYChart.Series<Number, Number> points = new XYChart.Series<>();
		points.setName("Included");
		points.getData().addAll(w.getPoints().stream().map(point -> createData(point)).collect(Collectors.toList()));
		scatter.getData().add(points);
		
		offlineList = new ArrayList<>();
		

		w.getPoints().addListener((ListChangeListener<? super WavefrontPoint>) e -> {
			while (e.next()) {
				if (e.wasAdded()) {
					List<XYChart.Data<Number, Number>> xyList = e.getAddedSubList().stream().map(point -> createData(point)).collect(Collectors.toList());
					synchronized(offlineList) {
						offlineList.addAll(xyList);
					}
//					Platform.runLater(() -> points.getData().addAll(xyList));
				}
			}
		});

		w.getFailedCaptures().addListener((ListChangeListener<? super PointLocation>) e -> {
			while (e.next()) {
				if (e.wasAdded()) {
					List<XYChart.Data<Number, Number>> xyList = e.getAddedSubList().stream().map(point -> createData(point)).collect(Collectors.toList());
					synchronized(offlineList) {
						offlineList.addAll(xyList);
					}
//					Platform.runLater(() -> points.getData().addAll(xyList));
				}
			}
		});
		w.processing().addListener((p, o, n) -> {
			if(!n) {
				System.out.println("Updating wavefront...");
				Platform.runLater(() -> {
					points.getData().clear();
					if(offlineList.contains(null)) {
						System.out.println("Offline list contains null!");
					}
					System.out.println("Adding offline list with "+offlineList.size()+" entries.");
					points.getData().addAll(offlineList);
					offlineList.clear();
				});
			}
		});
	}

	private XYChart.Data<Number, Number> createData(WavefrontPoint point) {
		XYChart.Data<Number, Number> item = new XYChart.Data<Number, Number>(point.getLocation().getX(),
				point.getLocation().getY());
		Circle circ = new Circle();
		circ.setFill(color(point.getCorrelationQuality()));
		circ.radiusProperty().bind(radius);
		item.setNode(circ);
		PointLocation loc = point.getLocation();
		setup(circ, loc,
				"Capture " + loc.getCaptureIndex() + "\nX: " + roundCM(loc.getX()) + " cm\nY: " + roundCM(loc.getY())
						+ "cm\nQuality: " + point.getCorrelationQuality() + "\n" + "Trigger: "
						+ loc.getCapture().getWaitTime() + " s");
		return item;
	}

	private Paint color(double correlationQuality) {
		double min = minQuality.get();
		double fac = (correlationQuality-min) / (1-min);
		return new Color(0, 1 - fac, fac, 1);
	}

	private XYChart.Data<Number, Number> createData(PointLocation loc) {
		XYChart.Data<Number, Number> item = new XYChart.Data<Number, Number>(loc.getX(), loc.getY());
		Circle circ = new Circle();
		circ.setFill(Color.BLACK);
		circ.radiusProperty().bind(radius);
		item.setNode(circ);
		setup(circ, loc, "Capture " + loc.getCaptureIndex() + "\nX: " + roundCM(loc.getX()) + " cm\nY: "
				+ roundCM(loc.getY()) + "cm\n" + "Trigger: " + loc.getCapture().getWaitTime() + " s");
		return item;
	}

	private void setup(Node node, PointLocation loc, String tooltip) {
		node.setOnMouseClicked(e -> {
			handler.showRaw(loc.getCaptureIndex());
		});
		node.setCursor(Cursor.HAND);
		
		tooltipMap.put(node, tooltip);
//
//		Tooltip t = new Tooltip(tooltip);
//
//		node.setOnMouseEntered(e -> {
//			Point2D point = node.localToScreen(0, 0);
//			t.show(node, point.getX() + 20, point.getY());
//		});
//		node.setOnMouseExited(e -> {
//			t.hide();
//		});
	}

	private static int roundCM(double meters) {
		return (int) Math.round(meters * 100);
	}

	public ObservableObjectValue<Wavefront> getWavefront() {
		return wavefront;
	}

	public SimpleDoubleProperty getRadius() {
		return radius;
	}

	public ScatterChart<Number, Number> getScatter() {
		return scatter;
	}

	public NumberAxis getxAxis() {
		return xAxis;
	}

	public NumberAxis getyAxis() {
		return yAxis;
	}

	public RequestHandler getHandler() {
		return handler;
	}

}
