import networkx as nx 
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import corner

class INARAVisualization:
    """Advanced visualization and reporting for INARA"""
    
    def __init__(self, config):
        self.config = config
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_spectral_fit(self, observed_spectrum, model_spectrum, wavelengths, uncertainties=None):
        """Plot observed vs model spectrum with residuals"""
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        
        ax0 = plt.subplot(gs[0])
        ax0.plot(wavelengths, observed_spectrum, 'k-', label='Observed', linewidth=2)
        ax0.plot(wavelengths, model_spectrum, 'r-', label='Model', linewidth=1.5)
        
        if uncertainties is not None:
            ax0.fill_between(wavelengths, 
                            model_spectrum - uncertainties,
                            model_spectrum + uncertainties,
                            alpha=0.3, color='red', label='Uncertainty')
        
        ax0.set_ylabel('Normalized Flux')
        ax0.legend()
        ax0.grid(True, alpha=0.3)
        
        ax1 = plt.subplot(gs[1])
        residuals = observed_spectrum - model_spectrum
        ax1.plot(wavelengths, residuals, 'k-', linewidth=1)
        ax1.axhline(0, color='red', linestyle='--', alpha=0.7)
        
        if uncertainties is not None:
            ax1.fill_between(wavelengths, 
                            -uncertainties,
                            uncertainties,
                            alpha=0.3, color='red')
        
        ax1.set_xlabel('Wavelength (μm)')
        ax1.set_ylabel('Residuals')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_corner(self, samples, param_names, truths=None):
        """Create corner plot of parameter posterior distributions"""
        fig = corner.corner(samples, labels=param_names, truths=truths,
                           quantiles=[0.16, 0.5, 0.84], show_titles=True,
                           title_kwargs={"fontsize": 12})
        return fig
    
    def plot_parameter_recovery(self, true_params, pred_params, param_names):
        """Plot parameter recovery accuracy"""
        fig, axes = plt.subplots(3, 4, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (true, pred, name) in enumerate(zip(true_params.T, pred_params.T, param_names)):
            axes[i].scatter(true, pred, alpha=0.6)
            axes[i].plot([true.min(), true.max()], [true.min(), true.max()], 'r--')
            axes[i].set_xlabel('True ' + name)
            axes[i].set_ylabel('Predicted ' + name)
            axes[i].grid(True, alpha=0.3)
            
            # Calculate and display R² score
            r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)
            axes[i].set_title(f'{name} (R² = {r2:.3f})')
        
        plt.tight_layout()
        return fig
    
    def plot_reaction_network(self, G, highlight_molecules=None):
        """Visualize reaction network"""
        plt.figure(figsize=(15, 12))
        
        # Use spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        node_colors = []
        for node in G.nodes():
            if highlight_molecules and node in highlight_molecules:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
                
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.4)
        
        # Draw labels for important nodes
        important_nodes = [node for node in G.nodes() if G.degree(node) > 2]
        labels = {node: node for node in important_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("Chemical Reaction Network")
        plt.axis('off')
        return plt.gcf()
    
    def create_summary_report(self, results, output_path):
        """Create comprehensive PDF report"""
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(output_path) as pdf:
            # Title page
            plt.figure(figsize=(8.5, 11))
            plt.text(0.5, 0.5, 'INARA Analysis Report\n\n' +
                    f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n" +
                    f"Number of samples: {results.get('n_samples', 'N/A')}\n" +
                    f"Average R² score: {results.get('avg_r2', 0):.3f}",
                    ha='center', va='center', size=16)
            plt.axis('off')
            pdf.savefig()
            plt.close()
            
            # Spectral fit
            if 'spectral_fit' in results:
                fig = self.plot_spectral_fit(**results['spectral_fit'])
                pdf.savefig(fig)
                plt.close(fig)
            
            # Parameter recovery
            if 'parameter_recovery' in results:
                fig = self.plot_parameter_recovery(**results['parameter_recovery'])
                pdf.savefig(fig)
                plt.close(fig)
            
            # Corner plot
            if 'corner_plot' in results:
                fig = self.plot_corner(**results['corner_plot'])
                pdf.savefig(fig)
                plt.close(fig)
            
            # Reaction network
            if 'reaction_network' in results:
                fig = self.plot_reaction_network(**results['reaction_network'])
                pdf.savefig(fig)
                plt.close(fig)
            
            # Biosignature analysis
            if 'biosignature_analysis' in results:
                plt.figure(figsize=(8.5, 11))
                biosig_data = results['biosignature_analysis']
                plt.text(0.1, 0.9, 'Biosignature Analysis', size=16)
                
                y_pos = 0.8
                for key, value in biosig_data.items():
                    plt.text(0.1, y_pos, f'{key}: {value:.3f}', size=12)
                    y_pos -= 0.05
                
                plt.axis('off')
                pdf.savefig()
                plt.close()
