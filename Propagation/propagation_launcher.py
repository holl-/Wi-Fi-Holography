import sys
from propagation_application_wx import launch
import argparse

wavefront_file = sys.argv[1]
print wavefront_file

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--multithreaded', action="store_true")
parser.add_argument("-output", default="Propagation Diagrams/")
parser.add_argument("-dr", type=float, default=0.01)
parser.add_argument("-view", type=int, default=0)
parser.add_argument("-freq", type=int, default=-2)
parser.add_argument("--darkfield", action="store_true")
parser.add_argument("-source", type=int, default=223)
parser.add_argument("--save", action="store_true")


args, leftovers = parser.parse_known_args(sys.argv[2:])
print args

print "Available arguments: --multithreaded, -output, -dr, -view, -freq, --darkfield, -source"

launch(wavefront_file,
       save_dir=args.output, multithreaded=args.multithreaded, dr=args.dr,
       display_z=args.view, initial_frequency_index=args.freq, enable_darkfield=args.darkfield, source_z=args.source,
       save_diagram=args.save)
