# Nuclei Count vs Mask

Reproducibility code for the paper *"Image-Level Counts versus Instance Masks for
Nuclei Counting: An Equivalence Study in Histopathology"*.

This study compares two supervision regimes for nuclei counting in H&E-stained
histopathology images — image-level count labels versus full instance masks —
under one architecture (DSU-Net) and one training/evaluation protocol, using
equivalence testing to quantify the practical value of spatial supervision.

## Contents

All source code is in [`src/`](src/). See [`src/README.md`](src/README.md) for a
file-by-file mapping to each table and figure in the paper.

## Data

This study uses only public datasets: [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke),
[NuInsSeg](https://doi.org/10.5281/zenodo.10518968), and
[CoNIC](https://conic-challenge.grand-challenge.org/) (built from the Lizard collection).
No new data were generated.

## Model checkpoints

This repository contains source code only. Trained checkpoints (`.pt`) are
released separately; scripts that require them accept a path via `--ckpt`.
