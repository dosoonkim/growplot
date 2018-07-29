# growplot
Scripts for analyzing bacterial growth data.


# Current workflow

1. Run python data\_cleanup.py to process the xlsx
2. Run Rscript plot.R to process the resulting csv and plot it as svg
3. Run compose_figure.py to place the SVG along with several images into one figure
