import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from utils.preprocess_validation import calculate_metrics

class SchizophreniaEDA:
    def __init__(self, path_to_clinical_data = None, path_to_raw_images = None):
        """Initializes the EDA class with the dataset."""
        self.df = pd.read_csv(path_to_clinical_data)
        self.schiz_df = self._add_schiz_column()
        self._create_age_groups()

        self.path_to_raw_images = path_to_raw_images
    
    def _add_schiz_column(self):
        """Adds a binary schizophrenia column."""
        df = self.df.copy()
        df["schiz"] = df["dx"].apply(lambda x: "Schizophrenia" if "Schizophrenia" in x else "No Schizophrenia")
        return df
    
    def _create_age_groups(self):
        """Creates age group bins."""
        bins = [0, 20, 30, 40, 50, 60, float('inf')]
        labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69"]
        self.schiz_df["age_group"] = pd.cut(self.schiz_df["age"], bins=bins, labels=labels, right=False)
    
    def plot_age_distribution(self, save_path='schiz_age.png'):
        """Plots age group distribution by schizophrenia diagnosis."""
        age_group_dx_bin = self.schiz_df.groupby(["age_group", "schiz"], observed=False).size().unstack(fill_value=0)
        fig = age_group_dx_bin.plot(kind='bar', stacked=False, figsize=(10, 6))
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Diagnosis', bbox_to_anchor=[1,0.9])
        plt.tight_layout()
        plt.show()
        fig.get_figure().savefig(save_path)
    
    def plot_gender_distribution(self, save_path='schiz_gen.png'):
        """Plots schizophrenia diagnosis by gender."""
        schiz_bin_gender = self.schiz_df.groupby(["sex", "schiz"], observed=False).size().unstack(fill_value=0)
        fig = schiz_bin_gender.plot(kind='bar', stacked=False, colormap="Paired", figsize=(10, 6))
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(title='Diagnosis', bbox_to_anchor=[1,0.9])
        plt.tight_layout()
        plt.show()
        fig.get_figure().savefig(save_path)
    
    def get_age_statistics(self):
        """Returns age statistics for schizophrenia and non-schizophrenia groups."""
        df_schiz = self.schiz_df[self.schiz_df["schiz"] == "Schizophrenia"]
        df_ns = self.schiz_df[self.schiz_df["schiz"] == "No Schizophrenia"]
        return {
            "Schizophrenia": df_schiz["age"].describe(),
            "No Schizophrenia": df_ns["age"].describe()
        }
    
    def assess_raw_images(self):
        """Assesses the quality of raw images."""
        raw_metrics_all_images = {} # TODO: should contain paths to images

        # iterate over all .pt files in path_to_raw_images
        for path in self.path_to_raw_images:
            loaded_tensor = torch.load(path)
            tensor_to_numpy = loaded_tensor.numpy()
            raw_metrics_per_image = calculate_metrics(None, tensor_to_numpy)
            raw_metrics_all_images.append(raw_metrics_per_image)
        return raw_metrics_all_images
    
    def save_raw_images_metrics(self):
        """Saves raw image metrics to a CSV file."""
        raw_metrics_all_images = self.assess_raw_images()
        df_raw_metrics = pd.DataFrame(raw_metrics_all_images)
        df_raw_metrics.to_csv('raw_image_metrics.csv', index=False)