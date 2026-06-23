## JAGO v0

JAGO converts H&E whole-slide images into spatial cell graphs for modeling tumor tissue architecture.

### Current pipeline

1. Run Hover-Net on TCGA-BRCA H&E whole-slide images.
2. Parse Hover-Net JSON into a cell table.
3. Convert pixel coordinates into microns using slide metadata.
4. Build a 50 µm radius graph where nodes are cells and edges connect nearby cells.
5. Split the whole-slide graph into local 500 µm graph patches.
6. Compute patch-level architecture statistics.
7. Rank patches by tumor-rich, stromal-rich, immune-rich, tumor-immune contact, tumor-stroma contact, immune-stroma contact, same-type clustering, and mixed architecture scores.
8. Visualize representative architecture patches.

### Current proof of concept

On one TCGA-BRCA H&E slide, JAGO generated:

- 141,521 cells
- 2,481,579 cell-cell proximity edges
- 50 local graph patches
- Patch-level architecture rankings
- A representative figure panel of distinct tumor tissue architectures

### Core idea

JAGO treats tumor tissue not as pixels alone, but as a graph of cells and spatial relationships. This lets us quantify architecture such as tumor clustering, stromal organization, immune localization, and tumor-immune contact from routine H&E slides.