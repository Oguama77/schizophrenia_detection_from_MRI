<h1 align="center">Schizophrenia Classification from Magnetic Resonance Images</h1>
### Overview
This project focuses on exploring the effect of various preprocessing and data augmentation methods on the accuracy of schizophrenia detection from Magnetic Resonance (MR) images. The aim of this exploration is to address the existing ambiguity in how MR images are preprocessed before training, as most authors in literature perform preprocessing in MATLAB without clear instructions for reproducibility.

### Project Data:
Find the raw data at [SchizConnect](http://schizconnect.org/).  
The data comes as 3-dimensional MRI volumes in NifTI format



<p align="center">
  <img src="paper/figs/sample_volume.png" alt="Sample volume">
</p>


### Project Workflow:
Data Exploration >> [Preprocessing](src/utils/preprocess.py) & [Augmentation](src/augmentation.py) >> [Feature Extraction & Training](src/models/models.py) >> Visualizing training results

### Repository Structure
```.
|   .gitignore
|   pyproject.toml
|   README.md
|   schizo.code-workspace
|   selected_files.csv
|   tox.ini
|
+---src
|   |   augmentation_pipeline.py
|   |   best_pipeline.py
|   |   create_raw_train_set.py
|   |   create_train_set.py
|   |   logger.py
|   |   main.py
|   |   preprocessing_pipeline.py
|   |   __init__.py
|   |
|   +---models
|   |       models.py
|   |       __init__.py
|   |
|   +---paper
|   |   |   paper.bbl
|   |   |   paper.bib
|   |   |   paper.pdf
|   |   |   paper.tex
|   |   |   preamble.tex
|   |   |
|   |   \---figs
|   |           empty.pdf
|   |           model_architecture.png
|   |
|   \---utils
|       |   augmentation.py
|       |   data_loader.py
|       |   preprocess.py
|       |   preprocess_validation.py
|       |   __init__.py
|       |
|       \---__pycache__
|               __init__.cpython-312.pyc
|
+---tests
|       test_data_loader.py
|       test_preprocess.py
|       __init__.py
|
\---tools
        convert_nii_pt.py
        sample_volume.png
```

## To run the project follow this commands:
All command should run under project root/working-directory
```bash 
#install Virtualenv is - a tool to set up your Python environments
pip install virtualenv
#create virtual environment (serve only this project):
python -m venv venv
#activate virtual environment
.\venv\Scripts\activate
+ (venv) should appear as prefix to all command (run next command just after activating venv)
#update venv's python package-installer (pip) to its latest version
python.exe -m pip install --upgrade pip
#install projects packages (Everything needed to run the project)
pip install -e .
#install dev packages (Additional packages for linting, testing and other developer tools)
pip install -e .[dev]
``` 