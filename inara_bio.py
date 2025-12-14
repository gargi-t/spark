import pandas as pd  # Add this line
import numpy as np 
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, rdFMCS, MolFromSmiles, MolToSmiles
from rdkit.Chem import rdChemReactions as Reactions
from rdkit.Chem.rdmolops import AddHs
# from rdkit.CML import CMLReact
import networkx as nx

class AdvancedBiosignatureAnalyzer:
    """Advanced cheminformatics for biosignature detection"""
    
    def __init__(self, config):
        self.config = config
        self.molecule_db = None
        self.reaction_db = None
        
    def load_databases(self, molecules_path, reactions_path):
        """Load molecular and reaction databases"""
        self.molecule_db = pd.read_csv(molecules_path)
        self.reaction_db = pd.read_csv(reactions_path)
        print(f"Loaded {len(self.molecule_db)} molecules and {len(self.reaction_db)} reactions.")
    
    def calculate_molecular_descriptors(self, smiles):
        """Calculate comprehensive molecular descriptors"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        descriptors = {
            'molecular_weight': Descriptors.MolWt(mol),
            'complexity': Descriptors.BertzCT(mol),
            'polar_surface_area': Descriptors.TPSA(mol),
            'logp': Descriptors.MolLogP(mol),
            'h_bond_donors': Descriptors.NumHDonors(mol),
            'h_bond_acceptors': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'ring_count': Descriptors.RingCount(mol),
            'molar_refractivity': Descriptors.MolMR(mol),
            'electrotopological_state': np.mean([Descriptors.EState_VSA1(mol), 
                                               Descriptors.EState_VSA2(mol)]),
            'synthetic_accessibility': self._calculate_synthetic_accessibility(mol)
        }
        return descriptors
    
    def _calculate_synthetic_accessibility(self, mol):
        """Estimate synthetic accessibility score"""
        # Simplified version of SA Score
        return min(1.0, Descriptors.MolWt(mol) / 500 + Descriptors.RingCount(mol) / 10)
    
    def find_biosignature_candidates(self, min_complexity=150, min_reactivity=0.5):
        """Find molecules with high biosignature potential"""
        candidates = []
        for _, row in self.molecule_db.iterrows():
            descriptors = self.calculate_molecular_descriptors(row['smiles'])
            if descriptors and (descriptors['complexity'] > min_complexity or 
                              descriptors['synthetic_accessibility'] > min_reactivity):
                biosignature_score = (
                    0.4 * descriptors['complexity'] / 200 +
                    0.3 * descriptors['synthetic_accessibility'] +
                    0.3 * (descriptors['h_bond_donors'] + descriptors['h_bond_acceptors']) / 10
                )
                candidates.append((row['name'], row['smiles'], descriptors, biosignature_score))
        
        return sorted(candidates, key=lambda x: x[3], reverse=True)
    
    def generate_reaction_network(self, core_molecules, max_depth=3):
        """Generate complex reaction network from core molecules"""
        G = nx.DiGraph()
        
        for mol in core_molecules:
            G.add_node(mol, type='molecule')
            
        # Simulate reactions (this would use real reaction rules in production)
        for depth in range(max_depth):
            new_nodes = []
            for node in list(G.nodes()):
                if G.nodes[node]['type'] == 'molecule':
                    # Find possible reactions for this molecule
                    possible_reactions = self._find_reactions_for_molecule(node)
                    
                    for rxn in possible_reactions:
                        products = self._apply_reaction(node, rxn)
                        for product in products:
                            if product not in G:
                                G.add_node(product, type='molecule')
                                new_nodes.append(product)
                            G.add_edge(node, product, reaction=rxn['name'])
            
            print(f"Depth {depth}: Added {len(new_nodes)} new molecules")
            
        return G
    
    def _find_reactions_for_molecule(self, molecule_smiles):
        """Find possible reactions for a given molecule"""
        # Placeholder for real reaction matching
        # This would use substructure matching in a real implementation
        mol = Chem.MolFromSmiles(molecule_smiles)
        possible_reactions = []
        
        # Simple reaction patterns
        if Chem.MolFromSmarts('[C]=[O]').HasSubstructMatch(mol):  # Carbonyl group
            possible_reactions.append({'name': 'reduction', 'type': 'redox'})
        if Chem.MolFromSmarts('[NH2]').HasSubstructMatch(mol):  # Amine group
            possible_reactions.append({'name': 'acylation', 'type': 'addition'})
            
        return possible_reactions
    
    def _apply_reaction(self, reactant_smiles, reaction):
        """Apply a reaction to a molecule"""
        # Placeholder for real reaction application
        reactant = Chem.MolFromSmiles(reactant_smiles)
        
        if reaction['name'] == 'reduction':
            # Simple reduction: C=O -> CH-OH
            product = Chem.MolFromSmiles(reactant_smiles.replace('=O', '[H]O'))
            return [Chem.MolToSmiles(product)] if product else []
        elif reaction['name'] == 'acylation':
            # Simple acylation: R-NH2 + R'-COOH -> R-NH-COR'
            return [reactant_smiles + '.C(=O)O']  # Placeholder
            
        return []
    
    def analyze_network_properties(self, reaction_network):
        """Analyze properties of the reaction network"""
        # Calculate network metrics
        degrees = dict(reaction_network.degree())
        betweenness = nx.betweenness_centrality(reaction_network)
        closeness = nx.closeness_centrality(reaction_network)
        
        # Find key molecules (hubs)
        hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Find potential biosignature pathways
        pathways = []
        for source in reaction_network.nodes():
            for target in reaction_network.nodes():
                if source != target:
                    try:
                        path = nx.shortest_path(reaction_network, source, target)
                        if 2 <= len(path) <= 5:  # Reasonable pathway length
                            pathways.append((source, target, path))
                    except nx.NetworkXNoPath:
                        pass
        
        return {
            'hubs': hubs,
            'betweenness_centrality': betweenness,
            'closeness_centrality': closeness,
            'pathways': pathways[:100]  # Limit to top 100 pathways
        }
    
    def detect_disequilibrium_patterns(self, atmospheric_params):
        """Detect chemical disequilibrium suggestive of life"""
        # Calculate thermodynamic disequilibrium
        # This would use actual thermodynamic calculations in production
        h2o, co2, ch4, co, nh3 = atmospheric_params[:5]
        
        # Simple redox disequilibrium indicators
        redox_ratio = (ch4 + nh3) / (co2 + co)
        oxidation_state = co2 - 0.5 * ch4 - 1.5 * nh3
        
        # Complexity metrics
        molecular_diversity = len([x for x in atmospheric_params[:8] if x > -6])
        
        return {
            'redox_disequilibrium': redox_ratio,
            'oxidation_state_anomaly': oxidation_state,
            'molecular_diversity': molecular_diversity,
            'biosignature_potential': 0.3 * redox_ratio + 0.4 * abs(oxidation_state) + 0.3 * molecular_diversity
        }