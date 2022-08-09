# DeepCoCoS code files

This repo contains all the code files used for the DeepCoCoS paper.
The repo is divided to subfolders according to the different languages.

The python code contain the DL pipeline for training a model for converting dispersed images to multi-channel images using the U-Net.
It also contains the code to convert the accession codes from the nCounter RCC files to their barcode color sequences using the RLF file.

The Matlab folder contains all post analysis including: barcode readout, statistical analysis and plots introduced in the text.
In order to run the GTPred_Comparison_Analysis in Matlab, the raw NanoString barcode stacks will be uploaded to an external file repository due to their size.

The ImageJ folder contains the script used to filter out-of-focus FOVs prior to the template matching barcode detection on the prediction and GT hyperstacks.
