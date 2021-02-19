# amusia_study

In this repository you will find the code used to run the experiment and analyze the data for an EEG project on mismatch negativity (MMN) responses in congenital amusia. The project addressed several questions. The main one, which is reported in study 1, was wheter amusic listeners are sensitive to the predictability of melodies. The stimuli were manipulated by changing their complexity and familiarity. Results informing other questions will be reported elsewhere. Data will be shared upon request.

The repository has the following content:

- "materials/MMN_paradigm/": The materials used to run the paradigms including matlab functions to randomize the stimuli taken from https://github.com/drqm/Musical-MMN-paradigms. We also include .sce and .pcl files to display the stimuli with Presentation (Neurobs), as well as the sound files used.

- "Preprocessing_scripts/": Here we include scripts to visualize, clean and average the data. Scripts include automatic and manual ICA as well as epoching and averaging of event related responses. Note that all the blocks included in the experiment (even those not reported in study 1) are processed with these scripts. Analyses of these data will be reported in the future.

- "Study1/": here we report the analysis scripts ("scripts/") and results ("results/") of study 1, focusing on the effect of complexity and familiarity on MMN responses in amusics. Python scripts for cluster-based analyses at the scalp level, as well as R scripts for mean amplitude analyses are included.

