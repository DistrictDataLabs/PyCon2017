#!/usr/bin/env python3
# figures
# Script to generate figures for the PyCon 2017 Yellowbrick Poster.
#
# Author:   Benjamin Bengfort <benjamin@bengfort.com>
# Created:  Wed May 10 17:26:09 2017 -0400
#
# Copyright (C) 2016 Bengfort.com
# For license information, see LICENSE.txt
#
# ID: figures.py [] benjamin@bengfort.com $

"""
Script to generate figures for the PyCon 2017 Yellowbrick Poster.
"""

##########################################################################
## Imports
##########################################################################

import os

import pandas as pd
import yellowbrick as yb
import matplotlib.pyplot as plt

# Default place to store figures directory
BASEDIR = os.path.dirname(__file__)
DATADIR = os.path.join(BASEDIR, "data")

# Listing of all the figures to generate
FIGURES = {

}

# Listing of datasets and their paths
datasets = {
    "bikeshare": os.path.join(DATADIR, "bikeshare", "bikeshare.csv"),
    "concrete": os.path.join(DATADIR, "concrete", "concrete.csv"),,
    "credit": os.path.join(DATADIR, "credit", "credit.csv"),,
    "energy": os.path.join(DATADIR, "energy", "energy.csv"),,
    "game": os.path.join(DATADIR, "game", "game.csv"),,
    "hobbies": os.path.join(DATADIR, "hobbies", "hobbies.csv"),,
    "mushroom": os.path.join(DATADIR, "mushroom", "mushroom.csv"),,
    "occupancy": os.path.join(DATADIR, "occupancy", "occupancy.csv"),,
}

##########################################################################
## Helper Functions
##########################################################################

def get_figures_path(name, ext=".pdf", root=BASEDIR):
    """
    Returns the path to store a figure with ``viz.poof(outpath=path)``,
    creating a figures directory if necessary.
    """
    # Location to store figures in
    figdir = os.path.join(root, "figures")

    # Create the figures directory if it does not exist
    if not os.path.exists(figdir):
        os.mkdir(figdir)

    # Determine if we need to add the default extension
    base, oext = os.path.splitext(name)
    if not oext:
        name = base + ext

    # Return the path to the figure
    return os.path.join(figdir, name)


def savefig(name, visualizer):
    """
    Saves the visualizer to the figures directory with the given name.
    """
    outpath = get_figures_path(name)
    visualizer.poof(outpath=outpath)


##########################################################################
## Data Loader
##########################################################################

# TODO: Add a data loader to cache datasets to reduce loading.

def load_data(name):
    # Get the path from the datasets
    path = datasets[name]

    # TODO: handle the NLP dataset
    # Return the data frame
    return pd.read_csv(path)


##########################################################################
## Visualizer Functions
##########################################################################



##########################################################################
## Main Method
##########################################################################

if __name__ == '__main__':
    # Render all uncommented figures
    for name, method in FIGURES.items():
        visualizer = method()
        savefig(name, visualizer)
