# RamanQualityControl
This tool can be used for fast and reliable quality assessment of Raman spectra. 

Two versions are available - a command line tool `spectra_qc.py` and an interactive IPython notebook `spectra_qc.ipynb`.  

The interactive version is used to test and fine-tune parameters before using the command line tool. It is recommended to only use ~20 spectra for this, as plots are genereated for each spectrum to help optimize the parameters, which could lead to slowdowns with many spectra.

The command line tool is used in the following way:  
`python spectra_qc.py [options] DIR`

DIR Path to the directory containing the spectra files. Spectra must be provided as data tables.

Options:

Flag | Description
--- | ---
-f, --filetype | File format/extension of the spectra files. Default: .txt  
-l, --limits | Lower and upper limits of the spectral range to be considered for quality assessment. Default: None None  
-p, --penalty | Penalty value used for smoothing the spectra. Higher --> stronger smoothing. Increase for noisy spectra. Default: 0  
-b, --buckets | Number of buckets used for subsampling. Recommended starting value is 1/10th the number of x-values. Default: 500  
-w, --halfwidth | Half width for the first iteration of peak suppression, in number of buckets. Recommended starting value is slightly wider than the widest peak. Default: 10  
-i, --iterations | Number of iterations of the peak suppression algorithm. Default: 5  
-W, --sgwindow | Window size used for smoothing by the Savitzky-Golay-Filter. Must be an odd integer. Default: 35  
-t, --threshold | Threshold value for the (negative) 2nd derivative below which a peak is recognized. Default: 0.5  
-s, --score | Intensity measure for calculating the quality score. 0: None, use 1 as the base score. 1: Median peak height; 2: Mean peak height; 3: Mean peak area; 4: Total peak area. Default: 1  
-n, --npeaks | How the number of peaks influences the score; 0: No influence; 1: Multiplicative, 2: Exponential. Default: 1  

The tool outputs a sorted list of the original spectra files with the total score and the number of peaks, as well as a plot showing the distribution of intensity and peak count.
Both the original and baseline corrected spectra pare sorted and provided in subfolders of the original spectra directory. 

A conda environment file with the necessary prerequisites is provided.

Example data can be found in the folder *test_spectra*.
