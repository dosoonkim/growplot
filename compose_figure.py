#!/usr/bin/env python3

"""
Compose a 'standard' figure for a ribosome deletion, combining 
growth rate plots with a 2D diagram and a pair of PyMOL depictions.

TODO: move past 'demo mode' and instead take a variable arg (for,
say, "construct" or "figure") on command line.

TODO: generalize formatting algorithm so that we can incorporate 
more or fewer figures as need be?
"""

#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import seaborn as sns
import svgutils
from svgutils.compose import *

# Hard to import PDF images back into python
only_fig = "Rplots.svg"
Figure("16cm", "16cm", 
    Panel(
        Image(240, 240, "2d_amw6.png"),
        #SVG(only_fig).scale(0.4),
        Text("A", 0, 15, size=12, weight='bold')
    ),
    Panel(
        #SVG(only_fig).scale(0.2),
        Image(150, 240, "overall.png"),
        Image(150, 240, "colored_merged.png").move(150,0),
        Text("B", 0, 15, size=12, weight='bold')
    ).move(280, 0),
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
