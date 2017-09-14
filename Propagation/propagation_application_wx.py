# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 20:23:07 2015

@author: Philipp
"""

from propagation_cuda import Propagator
import propagation_cuda

import tum_jet

import wavefront_util
import wavefront_format

import wx
 
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
import matplotlib.pyplot as plt

import numpy
import os
from scipy.optimize import curve_fit
import sys
import time
import threading
import ntpath
import pylab
import matplotlib.patches
from matplotlib.animation import FuncAnimation


 
class MatPlotLibCanvas(wx.Panel):
    def __init__(self,parent, dr, pos, size, dpi=120):
        wx.Panel.__init__(self, parent)
        self.figure = plt.figure(dpi=dpi)
        
        self.dr = dr
        self.pos = pos
        self.size = size
         
        self.canvas = FigureCanvas(self,-1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Hide()
        
        self.axes = self.figure.add_subplot(111)

        self.truth = None
        self.circle = None

        self.image = None
        self.use_tum_jet = True
        self.flip_ud = True

    def plot(self, image, use_tum_jet=True, flip_ud=True):
        self.image = image
        self.use_tum_jet = use_tum_jet
        self.flip_ud = flip_ud
        self.redraw()
     
    def redraw(self):
        self.axes.hold(False) # clears axes when next plot issued
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()

        # Draw image
        if self.image is not None:
            if self.flip_ud:
                image = numpy.flipud(self.image)

            pos_dr = (-self.pos[0]*self.dr, -self.pos[1]*self.dr)
            size_dr = (image.shape[0] * self.dr, image.shape[1] * self.dr)
            extent = (pos_dr[1],size_dr[1]+pos_dr[1], pos_dr[0], size_dr[0]+pos_dr[0])

            if self.use_tum_jet:
                self.axes.imshow(image, cmap=tum_jet.tum_jet, extent=extent)
            else:
                self.axes.imshow(image, extent=extent)

        # Set axes
        if xlim != (0.0, 1.0):
            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)
        self.axes.set_xlabel("width (m)")
        self.axes.set_ylabel("height (m)")

        # Draw special objects
        if self.truth is not None:
            self.axes.add_artist(plt.Circle(self.truth, 0.01, color="w"))

        if self.circle is not None:
            ellipse = matplotlib.patches.Ellipse((self.circle[0],self.circle[1]), width=self.circle[2], height=self.circle[3], fill=False, color="w", linewidth=4)
            self.axes.add_patch(ellipse)

        self.canvas.draw()

    def set_true_emitter(self, position_meters):
        self.truth = position_meters
        self.redraw()

    def add_circle(self, x, y, rad_x, rad_y):
        self.circle = (x,y,rad_x, rad_y)
        self.redraw()

    def plot_external(self, image, savefile, use_tum_jet=True, flip_ud=True, fontsize=8):
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()

        if flip_ud:
            image = numpy.flipud(image)

        fig = plt.figure(figsize = (1.7, 1.4175))

        pos_dr = (-self.pos[0] * self.dr, -self.pos[1] * self.dr)
        size_dr = (image.shape[0] * self.dr, image.shape[1] * self.dr)
        extent = (pos_dr[1], size_dr[1] + pos_dr[1], pos_dr[0], size_dr[0] + pos_dr[0])

        if use_tum_jet:
            plt.imshow(image, cmap=tum_jet.tum_jet, extent=extent)
        else:
            plt.imshow(image, extent=extent)
        if xlim != (0.0, 1.0):
            plt.xlim(xlim)
            plt.ylim(ylim)
        plt.xlabel("x", fontsize=fontsize)
        plt.ylabel("y", fontsize=fontsize)
        plt.xticks(xlim, fontsize=fontsize)
        plt.yticks(ylim, fontsize=fontsize)
        plt.subplots_adjust(top=1, bottom=.1, left=.14, right=.96)
        plt.savefig(savefile+".pdf", transparent=True, dpi=1200)
        plt.savefig(savefile+".png", transparent=True, dpi=1200)
        fig.show()

    def pixel_to_meters(self, x, y, image_shape):
        pos_dr = (-self.pos[0]*self.dr, -self.pos[1]*self.dr)
        size_dr = (image_shape[0] * self.dr, image_shape[1] * self.dr)
        extent = (pos_dr[1],size_dr[1]+pos_dr[1], pos_dr[0], size_dr[0]+pos_dr[0])
        x_rel = x / float(image_shape[1])
        y_rel = y / float(image_shape[0])
        x_m = x_rel * extent[0] + (1-x_rel) * extent[1]
        y_m = y_rel * extent[2] + (1-y_rel) * extent[3]
        return x_m, y_m

    def x_axis(self, array_length):
        pos_dr = (-self.pos[0] * self.dr, -self.pos[1] * self.dr)
        extent = (pos_dr[1], array_length * self.dr + pos_dr[1])
        return numpy.linspace(extent[0], extent[1], array_length)

    def y_axis(self, array_length):
        pos_dr = (-self.pos[0] * self.dr, -self.pos[1] * self.dr)
        extent = (pos_dr[0], array_length * self.dr + pos_dr[0])
        return numpy.linspace(extent[0], extent[1], array_length)


class WXPropagationFrame(wx.Frame):
    def __init__(self, parent, title, propagators, wf_offset, wf_size, save_dir, prefix, multithreaded, pinhole_rad, max_dist_cm, size=(900, 900)):
        wx.Frame.__init__(self,parent,title=title,size=size, style=wx.MINIMIZE_BOX|wx.SYSTEM_MENU|
                  wx.CAPTION|wx.CLOSE_BOX|wx.CLIP_CHILDREN)
        
        self.propagators = propagators
        self.save_dir = save_dir
        self.prefix = prefix
        self.multithreaded = multithreaded
        self.dr = propagators[0].get_dr()
        self.wf_offset = wf_offset
        self.wf_size = wf_size
        self.pinhole_rad = pinhole_rad

        self.darkfield_propagators = None
        self.pinhole_propagators = None
        self.darkfield_display = "Shadow"
        self.darkfield_clip = False

        self.distance = 0
        self.mode = 'Amp'  # 'Amp', 'HSV'
        self.frequency = len(propagators)/2
        self.direction = +1  # -Z
        self.mix = "RMS"

        self.wavefront = None
        
        self.sp = wx.SplitterWindow(self)
        self.canvas = MatPlotLibCanvas(self.sp, self.dr, wf_offset, wf_size)
        self.p2 = wx.Panel(self.sp,style=wx.SUNKEN_BORDER)
         
        self.sp.SplitHorizontally(self.canvas, self.p2, size[1]-230)
 
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText("Initialized")
        
        self.hibut = wx.Button(self.p2,-1,"Pan", size=(40,20),pos=(20,6))
        self.hibut.Bind(wx.EVT_BUTTON,self.pan)
        self.sibut = wx.Button(self.p2,-1,"Zoom", size=(40,20),pos=(60,6))
        self.sibut.Bind(wx.EVT_BUTTON,self.zoom)
        self.hmbut = wx.Button(self.p2,-1,"Home", size=(40,20),pos=(100,6))
        self.hmbut.Bind(wx.EVT_BUTTON,self.home)
         
        self.hibut = wx.Button(self.p2,-1,"Save diagram", size=(70,20),pos=(200,6))
        self.hibut.Bind(wx.EVT_BUTTON,self.save)
        self.hibut = wx.Button(self.p2,-1,"Save image", size=(70,20),pos=(200+70,6))
        self.hibut.Bind(wx.EVT_BUTTON,self.save_borderless)
        self.hibut = wx.Button(self.p2, -1, "Save print", size=(70, 20), pos=(200+70*2, 6))
        self.hibut.Bind(wx.EVT_BUTTON, self.show_print_size)
        self.hibut = wx.Button(self.p2, -1, "Animation", size=(70, 20), pos=(200+70*3, 6))
        self.hibut.Bind(wx.EVT_BUTTON, self.animate_up_to_current)

        self.cpub = wx.Button(self.p2,-1,"CPU", size=(40,20),pos=(505,10))
        self.cpub.Bind(wx.EVT_BUTTON,self.switchCPU)
        self.cudab = wx.Button(self.p2,-1,"CUDA", size=(40,20),pos=(505,30))
        self.cudab.Bind(wx.EVT_BUTTON,self.switchCuda)
        self.benchb = wx.Button(self.p2,-1,"Benchmark", size=(40,20),pos=(505,50))
        self.benchb.Bind(wx.EVT_BUTTON,self.benchmark)
        if not propagation_cuda.CUDA_ENABLED:
            self.cudab.Disable()
            self.benchb.Disable()
        
        
        self.cudab = wx.Button(self.p2,-1,"Amp", size=(40,20),pos=(550,10))
        self.cudab.Bind(wx.EVT_BUTTON,self.showAmp)
        
        self.cpub = wx.Button(self.p2,-1,"Phi", size=(40,20),pos=(550,30))
        self.cpub.Bind(wx.EVT_BUTTON,self.showPhase)
        
        self.cpub = wx.Button(self.p2,-1,"HSV", size=(40,20),pos=(550,50))
        self.cpub.Bind(wx.EVT_BUTTON,self.showHSV)

        self.rmsb = wx.Button(self.p2,-1,"RMS", size=(40,20),pos=(550,80))
        self.rmsb.Bind(wx.EVT_BUTTON,self.mix_rms)

        self.linb = wx.Button(self.p2, -1, "Lin", size=(40, 20), pos=(550, 100))
        self.linb.Bind(wx.EVT_BUTTON, self.mix_linear)

        self.negz = wx.Button(self.p2,-1,"-Z", size=(25,20),pos=(20,90))
        self.negz.Bind(wx.EVT_BUTTON, self.propagate_negative_z)

        self.posz = wx.Button(self.p2, -1, "+Z", size=(25, 20), pos=(50, 90))
        self.posz.Bind(wx.EVT_BUTTON, self.propagate_positive_z)

        self.locsrc = wx.Button(self.p2, -1, "Loc2D", size=(40, 20), pos=(260, 90))
        self.locsrc.Bind(wx.EVT_BUTTON, self.locate_source)
        self.locsrc = wx.Button(self.p2, -1, "Loc3D", size=(40, 20), pos=(300, 90))
        self.locsrc.Bind(wx.EVT_BUTTON, self.locate_source3d)

        self.darkb = wx.Button(self.p2, -1, "Darkfield", size=(90, 20), pos=(260+95, 90))
        self.darkb.Bind(wx.EVT_BUTTON, self.darkfield)

        self.darkinfo = wx.Button(self.p2, -1, "L/S", size=(40, 20), pos=(260+95+95, 90))
        self.darkinfo.Bind(wx.EVT_BUTTON, self.switch_darkfield_display)

        self.slider = wx.Slider(self.p2, -1, self.distance, 0, max_dist_cm, (0, 40), (500, -1),
                               wx.SL_AUTOTICKS | wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider.Bind(wx.EVT_SLIDER, self.updateDistance)

        self.freq_slider = wx.Slider(self.p2, -1, self.frequency, -1, len(propagators)-1, (590, 4), (50, 130),
                               wx.SL_AUTOTICKS | wx.SL_VERTICAL | wx.SL_LABELS)
        self.freq_slider.Bind(wx.EVT_SLIDER, self.update_frequency)

         
    def zoom(self,event):
        self.statusbar.SetStatusText("Zoom")
        self.canvas.toolbar.zoom()
 
    def home(self,event):
        self.statusbar.SetStatusText("Home")
        self.canvas.toolbar.home()
         
    def pan(self,event):
        self.statusbar.SetStatusText("Pan")
        self.canvas.toolbar.pan()

    def output_basename(self, type):
        freq_str = " freq "+str(self.frequency) if self.frequency >= 0 else ""
        base_str = self.save_dir + self.prefix + type + " at " + str(self.propagation_z()) + freq_str
        if self.is_darkfield_active():
            return base_str + " darkfield "+str(self.darkfield_propagators[0].get_wavefront_z())+" "+self.darkfield_display
        else:
            return base_str
    
    def save_borderless(self, event):
        extent = self.canvas.axes.get_window_extent().transformed(self.canvas.figure.dpi_scale_trans.inverted())
        self.canvas.figure.savefig(self.output_basename("img") + ".png", bbox_inches=extent)
    
    def save(self, event):
        self.canvas.figure.savefig(self.output_basename("diagram") + ".png", transparent=True)
        self.canvas.figure.savefig(self.output_basename("diagram") + ".pdf")

    def mix_rms(self, event):
        self.mix = "RMS"
        self.update_wavefront()

    def mix_linear(self, event):
        self.mix = "Linear"
        self.update_wavefront()

    def propagate_negative_z(self, event):
        self.direction = -1
        self.update_wavefront()

    def propagate_positive_z(self, event):
        self.direction = 1
        self.update_wavefront()
    
    def switchCuda(self, event):
        for propagator in self.propagators:
            propagator.set_use_cuda(True)
        self.statusbar.SetStatusText("Switched to CUDA")
        
    def switchCPU(self, event):
        for propagator in self.propagators:
            propagator.set_use_cuda(False)
        self.statusbar.SetStatusText("Switched to CPU")

    def propagation_z(self):
        return self.distance * self.direction

    def benchmark(self, event):
        times_cpu = numpy.empty((10,), dtype=numpy.float64)
        times_gpu = numpy.empty((10,), dtype=numpy.float64)
        
        for i in xrange(len(times_cpu)):
            self.statusbar.SetStatusText("Benchmarking CPU, pass "+str(i+1))
            start_time = time.time()
            self.propagators[0].__propagate_cpu(self.propagattion_z())
            end_time = time.time()
            times_cpu[i] = end_time - start_time
        
        for i in xrange(len(times_gpu)):
            self.statusbar.SetStatusText("Benchmarking CUDA, pass "+str(i+1))
            start_time = time.time()
            self.propagators[0].__propagate_cuda(self.propagation_z())
            end_time = time.time()
            times_gpu[i] = end_time - start_time
        
        mean_cpu = numpy.mean(times_cpu)
        mean_gpu = numpy.mean(times_gpu)
        factor = mean_cpu / mean_gpu
        
        self.statusbar.SetStatusText("Mean CPU: "+str(int(round(mean_cpu*1000)))+
                " ms, Mean CUDA: "+str(int(round(mean_gpu*1000)))+" ms, Factor "+str(round(factor, 1)))
        
            
    
    def showAmp(self, event):
        self.mode = 'Amp'
        self.update_canvas()
    
    def showPhase(self, event):
        self.mode = 'Phi'
        self.update_canvas()
    
    def showHSV(self, event):
        self.mode = 'HSV'
        self.update_canvas()
    
    def updateDistance(self, event):
        selectedDistance = event.GetInt() * 0.01
        if selectedDistance == self.distance:
            return
        self.distance = selectedDistance
        self.update_wavefront()

    def update_frequency(self, event):
        selected_frequency = event.GetInt()
        if selected_frequency == self.frequency:
            return
        self.frequency = selected_frequency
        self.update_wavefront()

    def locate_source3d(self, event):
        distances = numpy.linspace(self.distance - 0.15, self.distance + 0.15, 20)
        sigmas_x = []
        sigmas_y = []
        precisions = []
        for z in distances:
            print "Locating source at z=",z
            self.show_direct(z)
            sigma_x, sigma_y = self.locate_source(None, plot=False)
            sigmas_x.append(sigma_x)
            sigmas_y.append(sigma_y)
            precisions.append(max(sigma_x, sigma_y))

        # inverse_best, inverse_best_e, best_z, best_z_e, z_precision, z_precision_e, fit_curve = self.fit_gauss(distances, precisions, numpy.argmax(precisions), 0)
        # print "Best precision: ",1/inverse_best," at z=",best_z,"+-",best_z_e," - Gauss precision=",z_precision

        pylab.close()
        pylab.figure()
        pylab.plot(distances, precisions, label="X,Y combined", linewidth=2)
        # pylab.plot(distances, fit_curve, label="gauss fit")
        pylab.plot(distances, sigmas_x, label="sigma X")
        pylab.plot(distances, sigmas_y, label="sigma Y")
        pylab.xlabel("z (m)")
        pylab.ylabel("precision (m)")
        pylab.legend()
        pylab.show()


    def locate_source(self, event, cut_threshold_frac=0.4, plot=True):
        max_y, max_x = numpy.unravel_index(abs(self.wavefront).argmax(), self.wavefront.shape)

        x_values = abs(self.wavefront[max_y, :])
        x_axis = self.canvas.x_axis(len(x_values))
        y_values = abs(self.wavefront[:, max_x])
        y_axis = self.canvas.y_axis(len(y_values))

        # Calculate intensity %
        rad_px = int(round(0.09 / self.dr))
        # print "Summing circle with radius (px)", rad_px
        intensity_rms = 0.0
        for y in xrange(- rad_px, rad_px):
            for x in xrange(- rad_px, rad_px):
                if x**2 + y**2 <= rad_px**2:
                    intensity_rms += abs(self.wavefront[max_y+y, max_x+x])**2
        total_intensity = numpy.sum(abs(self.wavefront)**2)

        ratio = intensity_rms / total_intensity
        print "Intensity (RMS) % =",ratio
        print intensity_rms
        print total_intensity

        # Fit gauss along x-axis
        fitx_a, a_e, fitx_x0, x0_e, fitx_sigma, s_e, fitx_curve = self.fit_gauss(x_axis, x_values, max_x, cut_threshold_frac)
        fity_a, a_e, fity_y0, x0_e, fity_sigma, s_e, fity_curve = self.fit_gauss(y_axis, y_values, max_y, cut_threshold_frac)

        if plot:
            self.canvas.add_circle(fitx_x0, fity_y0,fitx_sigma*numpy.sqrt(2),fity_sigma*numpy.sqrt(2))

            x_m = x_axis[max_x]
            y_m = y_axis[max_y]
            self.statusbar.SetStatusText("Maximum at x=%.3f, y=%.3f" % (x_m, y_m))
            pylab.close()
            pylab.figure()
            pylab.plot(x_axis, x_values, label="x scan")
            pylab.plot(y_axis, y_values, label="y scan")
            pylab.plot(x_axis, fitx_curve, "b+", label="Gaus fit x")
            pylab.plot(y_axis, fity_curve, "b+", label="Gaus fit y")
            pylab.xlabel("physical location (m)")
            pylab.legend()
            pylab.show()
        return fitx_sigma, fity_sigma

    def fit_gauss(self, x_axis, x_values, max_x, cut_threshold_frac):
        print "----- Fit -------"
        cut_threshold = cut_threshold_frac * x_values[max_x]
        print "Cut Threshold", cut_threshold

        left_cut = 0
        right_cut = len(x_values)-2
        where_cut = numpy.where(x_values < cut_threshold)[0]
        for i in xrange(len(where_cut)):
            if where_cut[i] > max_x:
                right_cut = where_cut[i]
                left_cut = where_cut[i - 1]
                break
        print "Left cut at x =", x_axis[left_cut]
        print "Right cut at x =", x_axis[right_cut]

        x_values_cut = x_values[left_cut:right_cut + 1]
        x_axis_cut = x_axis[left_cut:right_cut + 1]

        gaus_f = lambda x, a, x0, sigma: a * numpy.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
        p0 = (x_values[max_x], x_axis[max_x], 0.04)
        popt, pcov = curve_fit(gaus_f, x_axis_cut, x_values_cut, p0)
        fit_a, fit_x0, fit_sigma = popt

        gaus_fit_data = [gaus_f(x, *popt) for x in x_axis]
        return fit_a, pcov[0,0], fit_x0, pcov[1,1], fit_sigma, pcov[2,2], gaus_fit_data

    def is_darkfield_active(self):
        return self.darkfield_propagators is not None

    def darkfield(self, event):
        if self.is_darkfield_active():
            self.darkfield_propagators = None
            self.update_wavefront()
            self.statusbar.SetStatusText("Darkfield disabled.")
        else:
            max_y, max_x = numpy.unravel_index(abs(self.wavefront).argmax(), self.wavefront.shape)
            pinhole_rad_px = self.pinhole_rad / self.dr # 0.0002
            pinhole = [
                [numpy.exp((-(x - max_x)**2-(y - max_y)**2)/pinhole_rad_px**2) for x in xrange(self.wavefront.shape[1])]
                for y in xrange(self.wavefront.shape[0])]  # anti- dark field; pinhole to select router emission cone
            pinhole = numpy.array(pinhole)
            mask = 1.0-pinhole

            wavefronts_original = propagation_cuda.propagate_array(self.propagators, self.propagation_z(), self.multithreaded)
            wavefronts_darkfield = [wavefront * mask for wavefront in wavefronts_original]
            wavefronts_pinhole = [wavefront * pinhole for wavefront in wavefronts_original]


            self.darkfield_propagators = [Propagator(wavefronts_darkfield[i], self.dr, self.propagators[i].get_wavelength(),
                                                     self.propagators[i].get_use_cuda(), wavefront_z=self.propagation_z())
                                          for i in xrange(len(wavefronts_original))]
            self.pinhole_propagators = [
                Propagator(wavefronts_pinhole[i], self.dr, self.propagators[i].get_wavelength(),
                           self.propagators[i].get_use_cuda(), wavefront_z=self.propagation_z())
                for i in xrange(len(wavefronts_original))]

            self.update_wavefront()
            self.statusbar.SetStatusText("Darkfield enabled.")

    def switch_darkfield_display(self, event):
        if self.darkfield_display == "Shadow":
            self.darkfield_display = "Light"
        else:
            self.darkfield_display = "Shadow"
        self.update_wavefront()
        self.statusbar.SetStatusText("Switched to "+self.darkfield_display)

    def update_wavefront(self):
        
        self.statusbar.SetStatusText("d = "+str(self.propagation_z()))
        start_time = time.time()

        if not self.is_darkfield_active():
            if self.frequency == -1:
                wavefronts = propagation_cuda.propagate_array(self.propagators, self.propagation_z(), self.multithreaded)
                self.wavefront = wavefront_util.blend_wavefront_array(wavefronts)
            else:
                self.wavefront = self.propagators[self.frequency].propagate(self.propagation_z())

        else: # darkfield
            if self.frequency == -1:
                wavefronts_direct = propagation_cuda.propagate_array(self.propagators, self.propagation_z(), self.multithreaded)
                wavefronts_darkfield = propagation_cuda.propagate_array(self.darkfield_propagators,  self.propagation_z(), self.multithreaded)
                wavefronts_pinhole = propagation_cuda.propagate_array(self.pinhole_propagators, self.propagation_z(), self.multithreaded)

                if self.darkfield_display == "Light":
                    # Light
                    result = pylab.zeros_like(self.wavefront)
                    result_pinhole = pylab.zeros_like(self.wavefront)
                    for f in xrange(len(wavefronts_direct)):
                        result += abs(wavefronts_darkfield[f]) ** 2 - abs(wavefronts_direct[f]) ** 2
                        result_pinhole += abs(wavefronts_pinhole[f]) ** 2
                    pinhole_normalization = pylab.sum(result * result_pinhole) / pylab.sum(result_pinhole ** 2)
                    self.wavefront = pinhole_normalization * result_pinhole - result

                else:
                    # Shadow: first mix direct/darkfield pairs and add up results (better)
                    self.wavefront = numpy.zeros_like(self.wavefront)
                    for f in xrange(len(wavefronts_direct)):
                        self.wavefront += wavefront_util.sub_wavefronts(wavefronts_darkfield[f], wavefronts_direct[f], method=self.mix)
                    self.wavefront = abs(self.wavefront)

            else: # single frequency
                wavefront_direct = self.propagators[self.frequency].propagate(self.propagation_z())
                wavefront_darkfield = self.darkfield_propagators[self.frequency].propagate(self.propagation_z())
                wavefront_pinhole = self.pinhole_propagators[self.frequency].propagate(self.propagation_z())

                if self.propagation_z() == self.darkfield_propagators[0].get_wavefront_z():
                    self.wavefront = abs(wavefront_darkfield)

                else:
                    if self.darkfield_display == "Light":
                        # Light
                        result_pinhole = abs(wavefront_pinhole) ** 2
                        result = abs(wavefront_darkfield)**2 - abs(wavefront_direct)**2
                        pinhole_normalization = pylab.sum(result * result_pinhole) / pylab.sum(result_pinhole * result_pinhole)
                        self.wavefront = pinhole_normalization * result_pinhole - result
                    else:
                        # Shadow
                        self.wavefront = wavefront_util.sub_wavefronts(wavefront_darkfield, wavefront_direct, method=self.mix)

            # TODO clip circle if selected
            self.wavefront += self.wavefront.min() # since the darkfield result is not complex, it should not contain negative values.

        
        end_time = time.time()
        self.time_calc = end_time - start_time
        self.statusbar.SetStatusText("d = "+str(self.propagation_z())+", simulation: "+str(int(round(self.time_calc*1000)))+" ms")
        
        self.update_canvas()
        
    
    def update_canvas(self):
        start_time = time.time()
        if self.mode == 'Amp':
            self.canvas.plot(abs(self.wavefront))
        elif self.mode == 'HSV':
            print "HSV not supported"
            self.hsv = self.propagators.last_to_rgb(brightness=1)
            self.canvas.plot(self.hsv)
        elif self.mode == 'Phi':
            self.canvas.plot(numpy.angle(self.wavefront))
        
        end_time = time.time()
        self.statusbar.SetStatusText("d = "+str(self.propagation_z())+
                ", simulation: "+str(int(round(self.time_calc*1000)))+
                " ms, visualization: "+str(int(round((end_time - start_time)*1000)))+" ms.")

    def show_print_size(self, event):
        self.canvas.plot_external(abs(self.wavefront), self.output_basename("print"))

    def animate_up_to_current(self, event, frames=None):
        if frames == None:
            frames = int(round(self.distance * 50))

        anim_dir = self.save_dir + self.prefix + "anim "+str(self.distance)+"/"
        os.mkdir(anim_dir)
        print "Rendering %d frames to %s" % (frames, anim_dir)

        for i,z in enumerate(numpy.linspace(0, self.distance, frames)):
            self.show_direct(z)
            self.canvas.figure.savefig(anim_dir+ "frame"+ str(i).zfill(4) + ".png", transparent=True)

        # animation = FuncAnimation(self.canvas.figure, self.show_direct, frames=numpy.linspace(0, self.distance, 5), interval=200)
        # anim_path = self.save_dir+self.prefix+"animation_to_"+str(self.distance)+".gif"
        # animation.save(anim_path, dpi=80, writer='imagemagick')
        print "Saved animation to "+anim_dir


    def show_darkfield(self, source_z=223, display_z=101, frequency=-1):
        self.distance = source_z * 0.01
        self.update_wavefront()
        self.darkfield(None)
        self.frequency = frequency
        self.distance = display_z * 0.01
        self.slider.SetValue(display_z)
        self.freq_slider.SetValue(frequency)
        self.update_wavefront()

    def show_direct(self, display_z, frequency=None):
        self.darkfield_propagators = None
        if frequency is not None:
            self.frequency = frequency
            self.freq_slider.SetValue(frequency)
        self.distance = display_z
        self.slider.SetValue(display_z*100)
        self.update_wavefront()


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def create_propagator(frequency_index, propagators, wavefront, dr, normalize=True):
    outer_grid, offset, inner_shape = wavefront_util.interpolate_to_raster(wavefront, frequency_index=frequency_index,
                                                                           dr=dr, spacing=0.5)
    if normalize:
        outer_grid /= numpy.mean(abs(outer_grid))
    frequency = wavefront.get_frequencies()[frequency_index]
    wavelength = 299792458 / (frequency)
    print "Frequency at index", frequency_index, "is", (frequency / 1e9), "GHz with wavelength", wavelength, "m"
    sys.stdout.flush()
    prop = Propagator(outer_grid, dr, wavelength, use_cuda=False)
    propagators.append(prop)


def launch(wavefront_file, dr=0.01, max_dist=15, save_dir="../", multithreaded=True, pinhole_rad=0.0002, display_z=101, initial_frequency_index=-1, enable_darkfield=True, source_z=223, save_diagram=False, truth=None):
    name = path_leaf(wavefront_file)[0:-len(".csv")]
    print "Version 2016-11-28"
    print "The Working directory is "+os.getcwd()
    print "Output directory set to "+os.path.realpath(save_dir)
    print "Loading Wavefront "+name+" ..."
    wavefront = wavefront_format.load_wavefront(wavefront_file)
    sys.stdout.flush()
    print "Loaded Wavefront with",str(wavefront.size()),"entries"
    sys.stdout.flush()

    outer_grid, offset, inner_shape = wavefront_util.interpolate_to_raster(wavefront, frequency_index=0,
                                                                           dr=dr, spacing=0.5)
    print "Constructing outer grid with",outer_grid.shape[0],"x",outer_grid.shape[1],"."

    propagators = []
    threads = []
    for frequency_index in xrange(wavefront.frequency_count()):
        if multithreaded:
            thread = threading.Thread(target=create_propagator, args=(frequency_index, propagators, wavefront, dr))
            threads.append(thread)
            thread.start()
        else:
            create_propagator(frequency_index, propagators, wavefront, dr)

    if multithreaded:
        for thread in threads:
            thread.join()
            print thread,"finished"

    app = wx.App(redirect=False)
    frame = WXPropagationFrame(None, "Wavefront Propagation", propagators, offset, inner_shape,
                               save_dir, name+" ", multithreaded,
                               pinhole_rad, int(round(max_dist*100)))

    if truth is not None:
        frame.canvas.set_true_emitter(truth)


    # frame.show_direct()
    # hist_dc, edges_dc = numpy.histogram(abs(frame.wavefront).flatten(), bins=100, range=(0, abs(frame.wavefront).max()))
    #

    # hist_df, edges_df = numpy.histogram(abs(frame.wavefront).flatten(), bins=100, range=(0, abs(frame.wavefront).max()))
    #
    #
    # pylab.close()
    # pylab.figure()
    # pylab.plot(hist_dc, label="direct")
    # pylab.plot(hist_df, label="darkfield")
    # pylab.xlabel("Amplitude (a.u.)")
    # pylab.legend()
    # pylab.show()




    if initial_frequency_index < -1:
        initial_frequency_index = len(propagators) / 2

    if enable_darkfield:
        frame.show_darkfield(display_z=display_z, frequency=initial_frequency_index, source_z=source_z)
    else:
        frame.show_direct(display_z=display_z, frequency=initial_frequency_index)

    frame.Show()

    if save_diagram:
        frame.save(None)

    app.MainLoop()





#source = wavefront.create_point(1024, 512)
#source = wavefront.create_double_slit(1024, 512)
#source = wavefront.create_gaussian_sphere(1024, 512)
#source = wavefront.from_file("Wavefront_14.npy", dr=dr, spacing=2)
#launch(source, 0.01)

