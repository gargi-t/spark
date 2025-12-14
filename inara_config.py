
import os
import json
import yaml
import h5py
import numpy as np
from datetime import datetime
from astropy.io import fits

class INARAConfig:
    """Configuration management for the INARA project"""
    
    def __init__(self, config_path=None):
        self.default_config = {
            'data': {
                'n_samples': 50000,
                'n_wavelengths': 200,
                'wavelength_range': [0.5, 5.0],
                'molecular_species': ['H2O', 'CO2', 'CH4', 'CO', 'NH3', 'HCN', 'C2H2', 'TiO', 'VO'],
                'noise_levels': [0.0001, 0.001, 0.01],
                'train_test_split': 0.8,
                'validation_split': 0.1,
                'batch_size': 64,
                'cache_dir': './data/cache/',
                'output_dir': './results/'
            },
            'model': {
                'nn_layers': [512, 256, 128, 64],
                'dropout_rate': 0.3,
                'latent_dim': 16,
                'vae_beta': 0.001,
                'gan_noise_dim': 32,
                'learning_rate': 0.0005,
                'patience': 20
            },
            'physics': {
                't_range': [300, 3000],
                'p_range': [-6, 2],  # log10(p/bar)
                'radius_range': [0.8, 2.5],  # R_jup
                'mass_range': [0.1, 20.0],  # M_jup
                'max_iter': 1000,
                'tolerance': 1e-6
            },
            'cheminformatics': {
                'complexity_threshold': 150,
                'reactivity_threshold': 0.7,
                'similarity_threshold': 0.85,
                'max_path_length': 4
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = {**self.default_config, **yaml.safe_load(f)}
        else:
            self.config = self.default_config
            
        # Create directories
        os.makedirs(self.config['data']['cache_dir'], exist_ok=True)
        os.makedirs(self.config['data']['output_dir'], exist_ok=True)
        
    def save_config(self, path):
        """Save current configuration to file"""
        with open(path, 'w') as f:
            yaml.dump(self.config, f)
            
    def get_timestamp(self):
        """Get current timestamp for experiment tracking"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")