# Wi-Fi-Holography
Analysis software to create and back-propagate microwave light fields

The software consists of a Java application for wavefront creation and a Python program for back-propagating the wavefront.
Both Java 8 and a scientific distribution of Python 2 (like Anaconda) are required to run the software.

## Tutorial: Analyze the example data

There is one example recording included in this software distribution. It comes with the files `Example recording 2 GHz.rec.zip` which is the raw data and `Example recording 2 GHz_wavefront_5.csv`, an intermediate file containing the relative amplitudes and phases at every point.

To run the software, first install Java 8 or newer and Python 2 (e.g. through Anaconda). Then clone this repository and execute `Wavefron Viewer 1.1.2.jar`.

You should see a window showing the example recording and its properties inside a table.
Select this recording and select the tab "Raw". Here you can browse the individual recordings in the time and frequency domain.

The tab "Correlate" contains all the settings for processing the raw data. For the example file, enter a frequency offset of 2400 MHz, a frequency range from 100 to 120 MHz and a set the quality slider to 0.1. Make sure to press enter each time you enter a number in one of the fields. Then click "Create Wavefront".

After a short time the results are plotted in the "Wavefront" tab. You can click on the individual circles to jump to the raw data recorded at that point. The color indicates the signal strength at that point where blue means good signal and green indicates a bad reception. Black points are excluded from the analysis as the signal was too weak at that point. Which points are excluded can be controlled with the noise slider on the "Correlate" page.

To back-propagate the result, go to the "Propagate" tab. Select the appropriate file and click "Launch Propagation". This will execute the Python application.

The python Application allows you to view the wavefront at different depths. Use the horizontal slider to move the viewing plane in Z direction. Click the -Z and +Z buttons to switch between forward and backward propagation. The buttons "Amp" and "Phi" let you view the amplitude and phases, respecitvely. Use the vertical slider at the right to browse through the individual frequencies. If the slider is in position -1, all frequencies are calculated and mixed.