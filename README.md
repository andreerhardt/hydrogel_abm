In this Python software repository the code to reproduce data and images from the paper

<h1>Modeling cellular self-organization in strain-stiffening hydrogels</h1>

by

A. H. Erhardt, D. Peschka, C. Dazzi, L. Schmeller, A. Petersen, S. Checa, A. Münch, B. Wagner (2024). [10.1007/s00466-024-02536-7](https://doi.org/10.1007/s00466-024-02536-7)

is stored.

**Abstract:**  We derive a three-dimensional hydrogel model as a two-phase system of a fibre network and liquid solvent, where the
nonlinear elastic network accounts for the strain-stiffening properties typically encountered in biological gels. We use this
model to formulate free boundary value problems for a hydrogel layer that allows for swelling or contraction. We derive
two-dimensional plain-strain and plain-stress approximations for thick and thin layers respectively, that are subject to external
loads and serve as a minimal model for scaffolds for cell attachment and growth. For the collective evolution of the cells as
they mechanically interact with the hydrogel layer, we couple it to an agent-based model that also accounts for the traction
force exerted by each cell on the hydrogel sheet and other cells during migration. We develop a numerical algorithm for the
coupled system and present results on the influence of strain-stiffening, layer geometry, external load and solvent in/outflux
on the shape of the layers and on the cell patterns. In particular, we discuss alignment of cells and chain formation under
varying conditions.

**Dependencies:** Requires Python with legacy FEniCS 2019.1.0 and some standard libraries for plotting and data processing.

<h3>Hydrogel and cell coupling</h3>

Folder: `./abm_hydrogel`

Data: Generates data for Fig. 12,13,14,15,16 in the manuscript.

Usage: With each of the json files in the folder run the command 

`python3 main_ABM_Allen_Cahn_Hilliard.py JSONFILENAME.json`

to generate the corresponding data in an output folder. Then use the ... to generate the output.

<h3>ABM single and double cell interaction</h3>

Folder: `./abm_interaction`

Data: Generates data for Fig. 17,18,19 in the manuscript.

Usage: With each of the json files in the folder run the command 

`python3 main_interaction.py JSONFILENAME.json`

to generate the corresponding data or directly run the Linux `run0.sh` and `run1.sh` shell script. Similarly use 

`python3 main_plotter.py JSONFILENAME.json` and `python3 main_plotter_stretch.py JSONFILENAME.json` to generate the corresponding output files in the undeformed and deformed configuration, respectively.

<h3>Pure elastic material and hydrogel (no cells)</h3>

Folder: `./elastic_hydrogel`

Data: Generates data for Fig. 6,7,8,9 in the manuscript.

Usage: With each of the json files in the folder run the command 

`python3 main_stretching_Allen_Cahn_Hilliard.py JSONFILENAME.json`

to generate the corresponding output data. Then call `python3 plot_fig_{6,7,8,9}.py` to generate the corresponding plot in the manuscript.

<h3>Highly resolved single cells</h3>

Folder: `./highres_singlecell`

Data: Generates data for Fig. 20 in the manuscript.

Usage: With each of the json files in the folder run the command 

`python3 main_singlecell.py JSONFILENAME.json`

or the corresponding Linux shell script `run.sh`. Then run `python3 main_plothighlight.py JSONFILENAME.json` or the corresponding shell script `gen_plots.sh` to generate the figure.
