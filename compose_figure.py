#!/usr/bin/env python3

"""
Compose a 'standard' figure for a ribosome deletion, combining 
growth rate plots with a 2D diagram and a pair of PyMOL depictions.

TODO: move past 'demo mode' and instead take a variable arg (for,
say, "construct" or "figure") on command line.
toDONE

TODO: generalize formatting algorithm so that we can incorporate 
more or fewer figures as need be?
"""

#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import seaborn as sns
import svgutils
from svgutils.compose import *
import argparse

def make_figure(two_d_filename, three_d_filename, growth_file, score_vs_rmsd):
    # Hard to import PDF images back into python
    only_fig = "Rplots.svg"

    three_d_panel = None
    if len(three_d_filename) == 1: 
        three_d_panel = Panel(
            Image(150, 240, three_d_filename[0]),
            Image(150, 240, three_d_filename[1]).move(150,0),
            Text("B", 0, 15, size=12, weight='bold')).move(280, 0),
    elif len(three_d_filename) == 2:
        three_d_panel = Panel(
            Image(150, 240, three_d_filename[0]),
            Image(150, 240, three_d_filename[1]).move(150,0),
            Text("B", 0, 15, size=12, weight='bold')).move(280, 0),
    elif len(three_d_filename) == 3:
        three_d_panel = Panel(
            Image(150, 240, three_d_filename[0]),
            Image(150, 240, three_d_filename[1]).move(100,0),
            Image(150, 240, three_d_filename[2]).move(200,0),
            Text("B", 0, 15, size=12, weight='bold')).move(280, 0),
    else:
        print("Cannot support so many sub-panes:", len(three_d_filename))
        


    Figure("16cm", "16cm", 
        Panel(
            Image(240, 240, two_d_filename),#"2d_amw6.png"),
            #SVG(only_fig).scale(0.4),
            Text("A", 0, 15, size=12, weight='bold')
        ),
        three_d_panel,
        Panel(
            SVG(only_fig).scale(0.4),
            Text("C", 0, 15, size=12, weight='bold')
        ).move(0, 280),
        Panel(
            Image(300, 300, "forward_AMW6.png"),
            Text("D", 0, 15, size=12, weight='bold')
        ).move(280, 280)
    ).save("fig_final_compose.svg")


# Plan is: 
# +---------------+---------------+
# |               |               |
# |  2D diagram   |   PyMOL (1+)  |
# |               |               |
# |               |               |
# |               |               |
# +---------------+---------------+
# |               |               |
# |    Growth     |   score vs.   |
# |               |   RMSD (ff)   |
# |               |               |
# |               |               |
# +---------------+---------------+


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compose a 'standard' figure for a ribosome deletion, combining growth rate plots with a 2D diagram and one or more PyMOL depictions.")

    parser.add_argument('--two_d_filename', type=str, nargs=1, help='the raster image file for a 2D diagram to plot')
    parser.add_argument('--three_d_filename', type=str, nargs='+', help='one or more 3D depictions of the junction in question')
    parser.add_argument('--growth_file', type=str, nargs='+', help='growth plot SVG')
    # Enable SVG support sometime?
    parser.add_argument('--score_vs_rmsd', type=str, nargs=1, help='score vs. RMSD PNG')
    args = parser.parse_args()
    make_figure(two_d_filename=args.two_d_filename, three_d_filename=args.three_d_filename, growth_file=args.growth_file, score_vs_rmsd=args.score_vs_rmsd)


