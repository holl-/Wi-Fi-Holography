import pylab
import numpy

tum_raw = [
(  0, 101, 189),  #TUM Blue   
(065, 190, 255),  #TUM Light Blue 
(145, 172, 107),  #TUM Green     
#(181, 202, 130),  #TUM Light Green 
(255, 180,   0),  #TUM Yellow    
(255, 128,   0),  #TUM Orange   
(229, 052, 024),  #TUM Red       
(202, 033, 063)  #TUM Dark Red    
]

offsets = [.0, .35, .5,  .75, .85, .95, 1.0]

tum_colors = [( offsets[ci], (col[0]/255.0, col[1]/255.0, col[2]/255.0)) for ci, col in enumerate(tum_raw)]

tum_jet = pylab.matplotlib.colors.LinearSegmentedColormap.from_list("tum_jet", tum_colors)


diagram_colors = [tum_colors[0][1],
                tum_colors[2][1],
                tum_colors[6][1],
                tum_colors[3][1],
                tum_colors[1][1],
                tum_colors[5][1],
                tum_colors[4][1],
]

print tum_raw[4]

# xs = numpy.linspace(-1,1,101)

# xs, ys = pylab.meshgrid(xs,xs)

# fig=pylab.figure()
# fig.add_subplot(211)
# pylab.imshow(numpy.exp(-(xs*xs + ys*ys)/200.0))
# fig.add_subplot(212)
# pylab.imshow(numpy.exp(-(xs*xs + ys*ys)/200.0), cmap=tum_jet)
# pylab.show()

# fig=pylab.figure()
# fig.add_subplot(211)
# pylab.imshow(xs, aspect=.3)
# fig.add_subplot(212)
# pylab.imshow(xs, cmap=tum_jet, aspect=.3)
# pylab.show()

