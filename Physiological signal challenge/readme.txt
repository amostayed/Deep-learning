%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  The China Physiological Signal Challenge 2018: Automatic identification of the rhythm/morphology abnormalities in 12-lead ECGs. %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Second (final) Open-source Submission %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%  Multimedia and Augmented Reality Lab, College of Electrical Engineering and Computing Systems %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  University of Cincinnati, Cincinnati, Ohio, USA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Contributors: Ahmed Mostayed, Junye Luo, and Xingliang Shu %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Advisor: Dr. William G. Wee %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DIRECTORY STRUCTURE:
               Root
                  |
		              notebooks
			                  |
                        		  
		              codes
                        |
                        (*.py)
                  model (tensorflow chckpoint)
                        |
                        (*.index)
                        (*.data)
                        (*.meta)
		              training_set
                        |
	                      (*.mat)
                        (*.csv)
                  validation_set
                        |
	                      (*.mat)
                        (*.csv)
                  note.txt
                  cpsc2018.py
                  score_py3.py
                  README_en.txt
% 
% DEVELOPMENT ENVIRONMENT:
                 OS: Windows
                 Python: Anaconda 4.2.0 64-bit (Python 3.5.2)
                 Tensorflow GPU version
% REQUIREMENTS:
1. Python 3.x!
2. Dependencies: 
                 TensorFlow (use pip to install)
                 PyWavelets (https://pywavelets.readthedocs.io/en/latest/) (pip install PyWavelets)
                 Numpy
		             Scipy
		             glob
% INSTRUCTIONS:
1. Please put the evaluation data set (in .mat format) in the /validation_set directory
2. Name the annotation file for evaluation data "REFERENCE.csv" and put it in the /validation_set directory
3. Under Windows environment, run command: >python cpsc2018.py -p .\\validation_set\\
4. Under Windows environment, run command: >python score_py3.py -r .\\validation_set\\REFERENCE.csv
%
% Additional notes:
1. Must have TensorFlow and PyWavelets installed to work
2. Should work fine with the latest release of TensorFlow (tested on CPU version)
