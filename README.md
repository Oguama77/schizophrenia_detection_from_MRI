## Schizophrenia Classification from Magnetic Resonance Images

### Overview
This project focuses on exploring the effect of various preprocessing and data augmentation methods on the accuracy of schizophrenia detection from Magnetic Resonance (MR) images. The aim of this exploration is to address the existing ambiguity in how MR images are preprocessed before training, as most authors in literature perform preprocessing in MATLAB without clear instructions for reproducibility.

### Project Data
The raw data were obtained from the [SchizConnect](http://schizconnect.org/) database. The data obtained originates from the Center of Biomedical Research Excellence (COBRE) dataset in NifTI format and represents structural MR images. In the framework of the present study, the data was sampled to give a balanced representation of age and gender distributions in the dataset, prioritizing age distribution, which has been shown to affect model performance. Overall, the project includes data from 62 individuals, 30 with schizophrenia and 32 healthy controls.

Example of the raw data:

<p align="center">
  <img src="src/paper/figs/sample_volume.png" alt="Sample volume">
</p>

### Project Workflow
[Data Exploration](src/utils/data_visualization.py) >> [Preprocessing](src/utils/preprocess.py) & [Augmentation](src/utils/augmentation.py) >> [Feature Extraction & Training](src/models/models.py) >> [Visualizing training results](src/utils/model_plotter.py)

&check; Data Exploration: assessed demographic features and gender/age distributions;

&check; Preprocessing & Augmentation: experimented with different combinations of preprocessing and data augmentation techniques to determine which pipeline yields the best performance of the ML/DL models;

&check; Feature Extraction & Training: harnessed ResNet-18 to extract features and SVC to classify them;

&check; Visualizing: obtained accuracy metrics of the classifier.

#### Definitions and specifications

Parameters

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
|   |   logger.py
|   |   main.py
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
|       |   data_visualization.py
|       |   model_plotter.py
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
```

## Instructions to run the project:
All command should run under project root/working-directory
```bash 
# install Virtualenv is - a tool to set up your Python environments
pip install virtualenv
# create virtual environment (serve only this project):
python -m venv venv
# activate virtual environment
venv\Scripts\activate # Windows
source venv/bin/activate # Linux
+ (venv) should appear as prefix to all command (run next command just after activating venv)
# update venv's python package-installer (pip) to its latest version
python.exe -m pip install --upgrade pip
# install projects packages (Everything needed to run the project)
pip install -e .
# install dev packages (Additional packages for linting, testing and other developer tools)
pip install -e .[dev]
# specify the project configuration: edit the config file and save the configuration
nano config.yaml
# run the main script
python main.py
``` 