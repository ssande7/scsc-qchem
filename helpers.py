# -*- coding: utf-8 -*-

from rdkit import Chem
from rdkit.Chem import AllChem
from ipywidgets import widgets, interact, fixed, interactive
from IPython.display import display
import py3Dmol
import warnings
import numpy as np
import glob
import os
import uuid
from tempfile import mkdtemp
from shutil import rmtree
from debtcollector import moves

warnings.simplefilter('ignore')

class Psikit(object):
    def __init__(self, threads=4, memory=2800, debug=False):
        import psi4
        self.debug = debug
        self.psi4 = psi4
        if self.debug:
            self.psi4.core.set_output_file("psikit_out.dat", True)
        else:
            self.psi4.core.be_quiet()
        self.psi4.set_memory("{} MB".format(memory));
        #self.psi4.set_options({"save_jk": True})  # for JK calculation
        self.psi4.set_num_threads(threads);
        self.wfn = None
        self.mol = None
        self.pmol = None
        self.tempdir = mkdtemp()
        self.SMILES_was_input = False
        self.psi_optimized = False
        self.name = None
        self.fileprefix = None
        self.frequencies_calculated = False

    def __del__(self):
        self.clean()
        
    def clean(self):
        rmtree(self.tempdir)
    
    def read_from_smiles(self, smiles_str, opt=True):
        self.mol = Chem.MolFromSmiles(smiles_str)
        if opt:
            self.rdkit_optimize()   

    def rdkit_optimize(self, addHs=True):
        if addHs:
            self.mol = Chem.AddHs(self.mol)
        AllChem.EmbedMolecule(self.mol, useExpTorsionAnglePrefs=True,useBasicKnowledge=True)
        AllChem.UFFOptimizeMolecule(self.mol)

    def geometry(self, multiplicity=1, sym_tol=1e-3):
        xyz = self.mol2xyz(multiplicity=multiplicity)
        self.pmol = self.psi4.geometry(xyz)
        self.pmol.symmetrize(sym_tol)

    def energy(self, basis_sets= "hf3c/6-31+g**", return_wfn=True, multiplicity=1):
        self.geometry(multiplicity=multiplicity)
        scf_energy, wfn = self.psi4.energy(basis_sets, return_wfn=return_wfn)
        self.psi4.core.clean()
        self.wfn = wfn
        self.mol = self.xyz2mol()
        return scf_energy

    def set_name(self, name=None):
        if not self.name:
            if not name:
                name = uuid.uuid4().hex
            self.name = name
            self.psi4.core.IO.set_default_namespace(self.name)
            self.psi4.set_options({"writer_file_label": self.tempdir+'/'+self.name})
            self.fileprefix = self.psi4.core.get_writer_file_prefix(self.name)

    def optimize(self, basis_sets= "hf3c/6-31+g**", return_wfn=True, name=None, multiplicity=1, maxiter=50):
        self.set_name(name)
        self.geometry(multiplicity=multiplicity)
        self.psi4.set_options({'GEOM_MAXITER':maxiter})
        try:
            scf_energy, wfn = self.psi4.optimize(basis_sets, return_wfn=return_wfn)
            self.wfn = wfn
        except self.psi4.OptimizationConvergenceError as cError:
            print('Convergence error caught: {0}'.format(cError))
            self.wfn = cError.wfn
            scf_energy = self.wfn.energy()
            self.psi4.core.clean()
        self.mol = self.xyz2mol()
        self.psi_optimized = True
        return scf_energy

    def frequencies(self, basis_sets="scf/6-31g**", name=None, multiplicity=1, maxiter=50, write_molden_files=True):
        # Cheaper frequency calculation since accuracy not as important
        self.set_name(name)
        if not self.psi_optimized:
            print('Cannot calculate frequencies without first optimizing!')
            return
        self.geometry(multiplicity=multiplicity)
        self.psi4.set_options({'GEOM_MAXITER':maxiter})
        if write_molden_files:
            self.psi4.set_options({"normal_modes_write": True})
        try:
            # Don't need wfn. Updating self.wfn would change geometry from the more accurate earlier calculation
            scf_energy = self.psi4.frequencies(basis_sets, ref_gradient=self.wfn.gradient(), return_wfn=False)
        except self.psi4.OptimizationConvergenceError as cError:
            print('Convergence error caught: {0}'.format(cError))
            scf_energy = cError.wfn.energy()
            self.psi4.core.clean()
        self.mol = self.xyz2mol()
        self.frequencies_calculated = True
        return scf_energy

    def set_options(self, **kwargs):
        """
        http://www.psicode.org/psi4manual/1.2/psiapi.html
        IV. Analysis of Intermolecular Interactions
        and 
        http://forum.psicode.org/t/how-can-i-change-max-iteration-in-energy-method/1238/2
        """
        self.psi4.set_options(kwargs)

    def mol2xyz(self, multiplicity=1):
        return mol2xyz(self.mol)

    def xyz2mol(self, confId=0):
        natom = self.wfn.molecule().natom()
        mol_array_bohr = self.wfn.molecule().geometry().to_array()
        mol_array = mol_array_bohr * 0.52917721092
        nmol = Chem.Mol(self.mol)
        conf = nmol.GetConformer(confId)
        for i in range(natom):
            conf.SetAtomPosition(i, tuple(mol_array[i]))
        return nmol

    def print_text(self, text="blah"):
        print(text)
        return text
    
    def input_smiles(self, InputSMILES = "ClC(Cl)Cl"):
        mol = Chem.AddHs(Chem.MolFromSmiles(InputSMILES))
        print('Input molecule:')
        display(mol)
        print('Ready to continue!')
        self.mol = mol
        self.rdkit_optimize()
        self.SMILES_was_input = True
        return mol

    def widget_input_smiles(self, InputSMILES = "ClC(Cl)Cl"):
        text_widg = widgets.Text(value=InputSMILES,
                                 placeholder='Type something',
                                 description='Input Smiles:',
                                 disabled=False)
        inter_widg = interactive(self.input_smiles,
                           {'manual': True, 'manual_name': "Load Molecule", 'layout': "widgets.Layout(width='100px', height='auto')"},
                           i = text_widg)
        return inter_widg

    def initial_view(self, optimize = True, addHs = True):
        if optimize:
            self.rdkit_optimize(addHs = addHs)
        v = py3Dmol.view(width=300, height=300)
        v.addModel(Chem.MolToMolBlock(self.mol), 'mol')
        v.setStyle({'sphere':{'scale':0.40},'stick':{'radius':0.15}});
        v.setBackgroundColor('0xeeeeee');
        v.show()

    def widget_initial_view(self, optimize=True, addHs=True):
        button = widgets.Button(description="Show Minimized State", layout=widgets.Layout(width='200px', height='auto'))
        output = widgets.Output()
        def on_button_clicked(b):
            output.clear_output()
            with output:
                if self.SMILES_was_input:
                    self.initial_view(optimize, addHs)
                    self.print_chloroform_info()
                else:
                    print("No molecule input yet!")
        button.on_click(on_button_clicked)
        display(button, output)

    def print_chloroform_info(self):
        bohr2nm = 0.052918
        if self.mol.GetNumAtoms() != 5 or not self.mol.HasSubstructMatch(Chem.MolFromSmiles("ClC(Cl)Cl")):
            return
        psi4mol = self.wfn.molecule()
        H = None
        Cl1 = None
        Cl2 = None
        C = None
        for idx in range(5):
            sym = psi4mol.symbol(idx)
            match sym:
                case 'H':
                    H = idx
                case 'CL':
                    if Cl1 is None:
                        Cl1 = idx
                    else:
                        Cl2 = idx
                case 'C':
                    C = idx
            if H is not None and Cl1 is not None and Cl2 is not None and C is not None:
                break
        if H is None or Cl1 is None or Cl2 is None or C is None:
            # Should never be true
            return
        geom = psi4mol.geometry().np
        CH = geom[C] - geom[H]
        CCl1 = geom[C] - geom[Cl1]
        CCl2 = geom[C] - geom[Cl2]
        len_CH = np.linalg.norm(CH)
        len_CCl1 = np.linalg.norm(CCl1)
        len_CCl2 = np.linalg.norm(CCl2)
        print(f'\nBond Lengths:\n  C-H:\t\t{len_CH * bohr2nm:.4f} nm\n  C-Cl:\t\t{len_CCl1 * bohr2nm:.4f} nm')
        angle_HCCl = np.arccos(np.clip(np.dot(CH/len_CH, CCl1/len_CCl1), -1.0, 1.0)) * 180/np.pi
        angle_ClCCl = np.arccos(np.clip(np.dot(CCl1/len_CCl1, CCl2/len_CCl2), -1.0, 1.0)) * 180/np.pi
        print(f'\nBond Angles: \n  H-C-Cl:\t{angle_HCCl:.3f}°\n  Cl-C-Cl:\t{angle_ClCCl:.3f}°')

    def widget_run_quantum_optimization(self):
        button = widgets.Button(description="Run Quantum Optimization", layout=widgets.Layout(width='200px', height='auto'))
        output = widgets.Output()
        def on_button_clicked(b):
            output.clear_output()
            with output:
                if self.SMILES_was_input:
                    print('Running quantum optimization ... ')
                    self.optimize()
                    print('Done!')
                else:
                    print("No molecule input yet!")
        button.on_click(on_button_clicked)
        display(button, output)

    def widget_find_molecular_vibrations(self):
        button = widgets.Button(description="Find Molecular Vibrations", layout=widgets.Layout(width='200px', height='auto'))
        output = widgets.Output()
        def on_button_clicked(b):
            output.clear_output()
            with output:
                if self.SMILES_was_input:
                    print('Finding molecular vibrations ... ')
                    self.frequencies()
                    print('Done!')
                else:
                    print("No molecule input yet!")
        button.on_click(on_button_clicked)
        display(button, output)

    def show_normal_modes(self):
        """
        wrapper function that parses the file and initializes the widget.
        """
        if not self.frequencies_calculated:
            print('Molecular vibrations not yet calculated!')
            return
        all_frequencies, coords, normal_modes =  parse_molden(filename=self.fileprefix+'.molden_normal_modes')
        _ = interact(draw_normal_mode, coords=fixed(coords), normal_modes=fixed(normal_modes), mode = widgets.Dropdown(
            options=all_frequencies,
            value=0,
            description='Normal mode:',
            style={'description_width': 'initial'}
        ))

    def widget_show_normal_modes(self):
        button = widgets.Button(description="Show Vibrations", layout=widgets.Layout(width='200px', height='auto'))
        output = widgets.Output()
        def on_button_clicked(b):
            output.clear_output()
            with output:
                if self.SMILES_was_input:
                    self.show_normal_modes()
                else:
                    print("No molecule input yet!")
        button.on_click(on_button_clicked)
        display(button, output)

def input_smiles(InputSMILES="ClCCl"):
    mol = Chem.AddHs(Chem.MolFromSmiles(InputSMILES))
    print('Input molecule:')
    display(mol)
    print('Ready to continue!')
    return mol

def mol2xyz(mol, multiplicity=1):
    charge = Chem.GetFormalCharge(mol)
    xyz_string = "\n{} {}\n".format(charge, multiplicity)
    for atom in mol.GetAtoms():
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        xyz_string += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)
    return xyz_string


def section(fle, begin, end):
    """
    yields a section of a textfile. 
    Used to identify [COORDS] section etc
    """
    with open(fle) as f:
        for line in f:
            # found start of section so start iterating from next line
            if line.startswith(begin):
                for line in f: 
                    # found end so end function
                    if line.startswith(end):
                        return
                    # yield every line in the section
                    yield line.rstrip()    

def parse_molden(filename='default.molden_normal_modes'):
    """
    Extract all frequencies, the base xyz coordinates 
    and the displacements for each mode from the molden file
    """
    all_frequencies = list(section(filename, '[FREQ]', '\n'))
    all_frequencies = [("{:0.1f} THz".format(float(freq)*0.0299793),i) for i, freq in enumerate(all_frequencies)]
    # convert 1/cm to THz
    coords = list(section(filename, '[FR-COORD]', '\n'))
    normal_modes = []
    for freq in range(len(all_frequencies)):
        if freq+1 != len(all_frequencies):
            normal_modes.append(list(section(filename, f'vibration {freq+1}', 'vibration')))
        else:
            normal_modes.append(list(section(filename, f'vibration {freq+1}', '\n')))
    return all_frequencies, coords, normal_modes

def draw_normal_mode(mode=0, coords=None, normal_modes=None):
    """
    draws a specified normal mode using the animate mode from py3Dmol. 
    Coming from psi4 units need to be converted from a.u to A. 
    """
    fac=0.52917721067121  # bohr to A
    xyz =f"{len(coords)}\n\n"
    for i in range(len(coords)):
        atom_coords = [float(m) for m in  coords[i][8:].split('       ')]
        mode_coords = [float(m) for m in  normal_modes[mode][i][8:].split('       ')]
        xyz+=f"{coords[i][0:4]} {atom_coords[0]*fac} {atom_coords[1]*fac} {atom_coords[2]*fac} {mode_coords[0]*fac} {mode_coords[1]*fac} {mode_coords[2]*fac} \n"
    view = py3Dmol.view(width=300, height=300)
    view.addModel(xyz, "xyz", {'vibrate': {'frames':10,'amplitude':1}})
    view.setStyle({'sphere':{'scale':0.40},'stick':{'radius':0.15}})
    view.setBackgroundColor('0xeeeeee')
    view.animate({'loop': 'backAndForth'})
    view.zoomTo()
    return(view.show())



