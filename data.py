import os 
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from scipy import interpolate
import batman

class PhysicsSimulator:
    """Simulates physical processes for exoplanet atmospheres"""
    
    def __init__(self, config):
        self.config = config
        self.wavelengths = np.linspace(
            config['data']['wavelength_range'][0],
            config['data']['wavelength_range'][1],
            config['data']['n_wavelengths']
        )
        
    def radiative_transfer(self, params):
        """Simplified radiative transfer model with realistic physics"""
        # This would be replaced with petitRADTRANS/ExoTransmit in production
        n_samples = params.shape[0]
        spectra = np.ones((n_samples, len(self.wavelengths)))
        
        # Create absorption features for each molecule
        for i, species in enumerate(self.config['data']['molecular_species']):
            abundance = params[:, i]
            cross_section = self._molecular_cross_section(species)
            for j in range(n_samples):
                spectra[j] *= np.exp(-abundance[j] * cross_section)
                
        # Add gray cloud opacity
        n_species = len(self.config['data']['molecular_species'])
        cloud_params = params[:, n_species:n_species+2]

        if cloud_params.shape[1] >= 2:
            cloud_top_p = cloud_params[:, 0]
            cloud_thickness = cloud_params[:, 1]
        else:
            cloud_top_p = np.zeros(n_samples)
            cloud_thickness = np.zeros(n_samples)

        for j in range(n_samples):
            cloud_opacity = 0.1 * (10**cloud_top_p[j]) * cloud_thickness[j]
            spectra[j] *= np.exp(-cloud_opacity)
            
        return spectra
    
    def _molecular_cross_section(self, species):
        """Generate realistic molecular cross-sections"""
        # Placeholder for actual cross-section data
        if species == 'H2O':
            center, width, strength = 1.4, 0.1, 0.8
        elif species == 'CO2':
            center, width, strength = 2.0, 0.15, 0.7
        elif species == 'CH4':
            center, width, strength = 3.3, 0.2, 0.6
        else:
            center, width, strength = 2.5, 0.25, 0.5
            
        return strength * np.exp(-(self.wavelengths - center)**2 / (2 * width**2))
    
    def generate_transit_spectra(self, params, planet_params):
        """Generate transit depth spectra using batman"""
        n_samples = params.shape[0]
        transit_depths = np.zeros((n_samples, len(self.wavelengths)))
        
        for i in range(n_samples):
            for j, wl in enumerate(self.wavelengths):
                # Calculate radius based on atmospheric properties
                scale_height = 8000 * planet_params[i, 0] / planet_params[i, 1]  # T/g
                r_planet = planet_params[i, 2] + scale_height * np.log(1e6)  # Approximate
                
                # Use batman to generate transit model
                transit_params = batman.TransitParams()
                transit_params.t0 = 0.0
                transit_params.per = 1.0
                transit_params.rp = r_planet
                transit_params.a = 15.0
                transit_params.inc = 87.0
                transit_params.ecc = 0.0
                transit_params.w = 90.0
                transit_params.u = [0.1, 0.3]  # quadratic limb-darkening coefficients
                transit_params.limb_dark = "quadratic"
                
                # Create transit model
                t = np.linspace(-0.1, 0.1, 100)
                m = batman.TransitModel(transit_params, t)
                flux = m.light_curve(transit_params)
                
                # Store transit depth
                transit_depths[i, j] = 1 - np.min(flux)
                
        return transit_depths

class DataGenerator:
    """Generates and manages synthetic datasets for INARA"""
    
    def __init__(self, config):
        self.config = config
        self.physics_simulator = PhysicsSimulator(config)
        self.scaler = StandardScaler()
        self.feature_scaler = RobustScaler()
        
    def generate_parameters(self):
        """Generate physically plausible atmospheric parameters"""
        cfg = self.config['data']
        physics_cfg = self.config['physics']
        
        n_samples = cfg['n_samples']
        n_species = len(cfg['molecular_species'])
        
        # Molecular abundances (log scale)
        params = np.random.uniform(-8, -1, (n_samples, n_species))
        
        # Temperature and pressure parameters
        t_params = np.random.uniform(physics_cfg['t_range'][0], 
                                    physics_cfg['t_range'][1], 
                                    (n_samples, 2))
        
        p_params = np.random.uniform(physics_cfg['p_range'][0],
                                    physics_cfg['p_range'][1],
                                    (n_samples, 2))
        
        # Cloud parameters
        cloud_params = np.random.uniform(-3, 1, (n_samples, 2))
        
        # Planet physical parameters
        planet_params = np.column_stack([
            np.random.uniform(physics_cfg['t_range'][0], physics_cfg['t_range'][1], n_samples),  # Temperature
            np.random.uniform(8, 12, n_samples),  # log10(gravity)
            np.random.uniform(physics_cfg['radius_range'][0], physics_cfg['radius_range'][1], n_samples),  # Radius
            np.random.uniform(physics_cfg['mass_range'][0], physics_cfg['mass_range'][1], n_samples)   # Mass
        ])
        
        # Combine all parameters
        all_params = np.column_stack([params, t_params, p_params, cloud_params, planet_params])
        param_names = (
            cfg['molecular_species'] + 
            ['T_day', 'T_night'] + 
            ['log_P_base', 'log_P_top'] + 
            ['cloud_opacity', 'cloud_particle_size'] +
            ['planet_temp', 'log_g', 'planet_radius', 'planet_mass']
        )
        
        return all_params, param_names
    
    def generate_dataset(self, use_cache=True):
        """Generate or load the complete dataset"""
        cache_path = os.path.join(self.config['data']['cache_dir'], 'dataset.h5')
        
        if use_cache and os.path.exists(cache_path):
            print("Loading cached dataset...")
            with h5py.File(cache_path, 'r') as f:
                params = f['params'][:]
                spectra = f['spectra'][:]
                param_names = list(f['param_names'][:].astype(str))
            return params, spectra, param_names
        
        print("Generating new dataset...")
        params, param_names = self.generate_parameters()
        
        # Generate spectra using physics simulator
        n_species = len(self.config['data']['molecular_species'])
        atmospheric_params = params[:, :n_species + 6]  # molecules + T + P + clouds
        planet_params = params[:, -4:]  # planet physical parameters
        
        # Generate both emission and transit spectra
        emission_spectra = self.physics_simulator.radiative_transfer(atmospheric_params)
        transit_spectra = self.physics_simulator.generate_transit_spectra(atmospheric_params, planet_params)
        
        # Combine both types of spectra
        spectra = np.concatenate([emission_spectra, transit_spectra], axis=1)
        
        # Add realistic noise
        noise_level = np.random.choice(self.config['data']['noise_levels'], size=spectra.shape[0])
        for i in range(spectra.shape[0]):
            spectra[i] += np.random.normal(0, noise_level[i], spectra.shape[1])
        
        # Cache the dataset
        with h5py.File(cache_path, 'w') as f:
            f.create_dataset('params', data=params)
            f.create_dataset('spectra', data=spectra)
            f.create_dataset('param_names', data=np.array(param_names, dtype='S'))
        
        return params, spectra, param_names
    
    def prepare_training_data(self, params, spectra):
        """Prepare data for training with proper scaling and splitting"""
        # Scale spectra
        spectra_scaled = self.scaler.fit_transform(spectra)
        
        # Scale parameters
        params_scaled = self.feature_scaler.fit_transform(params)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            spectra_scaled, params_scaled, 
            test_size=1 - self.config['data']['train_test_split'],
            random_state=42
        )
        
        # Further split training into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config['data']['validation_split'] / self.config['data']['train_test_split'],
            random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
