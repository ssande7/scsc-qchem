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
        self.basis_set = 'hf3c/6-31+g**'

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

    def energy(self, return_wfn=True, multiplicity=1):
        self.geometry(multiplicity=multiplicity)
        scf_energy, wfn = self.psi4.energy(self.basis_set, return_wfn=return_wfn)
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

    def optimize(self, return_wfn=True, name=None, multiplicity=1, maxiter=50):
        self.set_name(name)
        self.geometry(multiplicity=multiplicity)
        self.psi4.set_options({'GEOM_MAXITER':maxiter})
        if self.mol.GetNumAtoms() < 6:
            self.basis_set = 'hf3c/6-31+G**'
        else:
            self.basis_set = 'hf/6-31G**'
        try:
            scf_energy, wfn = self.psi4.optimize(self.basis_set, return_wfn=return_wfn)
            self.wfn = wfn
        except self.psi4.OptimizationConvergenceError as cError:
            print('Convergence error caught: {0}'.format(cError))
            self.wfn = cError.wfn
            scf_energy = self.wfn.energy()
            self.psi4.core.clean()
        self.mol = self.xyz2mol()
        self.psi_optimized = True
        return scf_energy

    def frequencies(self, name=None, multiplicity=1, maxiter=50, write_molden_files=True):
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
            scf_energy = self.psi4.frequencies(self.basis_set, ref_gradient=self.wfn.gradient(), return_wfn=False)
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
        atom_coords = [float(m) for m in  coords[i][8:].split()]
        mode_coords = [float(m) for m in  normal_modes[mode][i][8:].split()]
        xyz+=f"{coords[i][0:4]} {atom_coords[0]*fac} {atom_coords[1]*fac} {atom_coords[2]*fac} {mode_coords[0]*fac} {mode_coords[1]*fac} {mode_coords[2]*fac} \n"
    view = py3Dmol.view(width=300, height=300)
    view.addModel(xyz, "xyz", {'vibrate': {'frames':10,'amplitude':1}})
    view.setStyle({'sphere':{'scale':0.40},'stick':{'radius':0.15}})
    view.setBackgroundColor('0xeeeeee')
    view.animate({'loop': 'backAndForth'})
    view.zoomTo()
    return(view.show())


import solara
import solara.lab

@solara.component
def Step1():
    with solara.Details(summary=solara.Markdown('### Step 1: Loading in a molecule'), expand=False):
        with solara.lab.Tabs():
            with solara.lab.Tab('Instructions'):
                solara.Markdown(
'''
We first need to tell the quantum chemistry software which molecule to look at, which we can do with a SMILES code.
"SMILES" stands for **simplified molecular-input line-entry system** and is a useful standard format for describing
molecules to simple computer programs - like this notebook!

The "InputSMILES" box below has the SMILES code for chloroform filled in by default -- when you click "Load Molecule",
you can check that the notebook displays a 2D structure of the correct molecule.

After you have completed the calculations for a chloroform molecule, you can look at the table in the next tab
("Some SMILES codes") and input SMILES codes for other common molecules, and then click "Load Molecule" to start
calculations for any of them.

If you'd like to look at other molecules not in that list, check out the "Building Molecules in Molview" tab
to see how to build molecules in Molview and obtain their SMILES codes!
''')
            with solara.lab.Tab('Some SMILES Codes'):
                with solara.Columns([1,1,2]):
                    with solara.Column():
                        solara.Markdown('#### Molecule')
                        solara.Text('Chloroform')        
                        solara.Text('Methane')           
                        solara.Text('Dichloromethane')   
                        solara.Text('Acetonitrile')      
                        solara.Text('Dimethylsulfoxide') 
                        solara.Text('Dimethylformamide') 
                        solara.Text('Benzene')           
                        solara.Text('Pyridine')          
                        solara.Text('Phenol')            
                        solara.Text('Aniline')           
                    with solara.Column():
                        solara.Markdown('#### SMILES')
                        solara.Text('ClC(Cl)Cl')
                        solara.Text('C')
                        solara.Text('ClCCl')
                        solara.Text('CC#N')
                        solara.Text('CS(=O)C')
                        solara.Text('CN(C)C=O')
                        solara.Text('C1=CC=CC=C1')
                        solara.Text('C1=CC=NC=C1')
                        solara.Text('C1=CC=C(C=C1)O')
                        solara.Text('C1=CC=C(C=C1)N')
                    solara.Column()
            with solara.lab.Tab('SMILES codes from Moleview'):
                solara.Markdown(
'''
If you'd like to look at a molecule that's not on the list, you can get a SMILES code from
[Molview](https://molview.org/):

<ol>
    <li>
    Once Molview has loaded, click on the "Clear all" button (with a _rubbish bin_ logo) to clear the screen and get ready for your own molecule!
    <ul>
        <li>The <i>white</i> area on the left will be cleared out. Don't worry about the <i>black</i> area on the right with a 3D molecular model - we will generate a new 3D model soon.</li>
    </ul>
    </li>

    <li>
    Now try building a chloroform molecule in the white area (if you get stuck, see instructions in the "(Molview) Building a chloroform molecule" tab!)
    <ul>
        <li><i><b>Hint:</b> For other molecules, if you're not sure how they look, try searching for their Lewis structure online, or looking them up by name with Molview's search bar.</i></li>
    </ul>
    </li>

    <li>
    Once you have built the molecule, here's how you get the SMILES code:
    <ol>
        <li>Look for the "Tools" dropdown menu, and then click on the "Information Card" in the middle of the menu.</li>
        <li>Look for an entry called "Canonical SMILES". You will see a series of upper-case letters, brackets, and equal-signs. That's the SMILES code!</li>
        <li>Copy it into the "InputSMILES" box in this notebook below, then click the "Load Molecule" button and check that the same molecule is loaded into the notebook.</li>
        <li>Now we can do some more accurate calculations on that molecule than what is possible with Molview.</li>
    </ol>
    </li>
</ol>
''')
            with solara.lab.Tab('Molview: Building a chloroform molecule'):
                solara.Markdown(
'''
Here's how you could build a chloroform molecule in Molview:

<ol>
    <li>Use "Clear all" to start again!</li>

    <li>
    The <i>right sidebar</i> has a list of elements to choose from.
    <ul>
        <li>Click on "C" (carbon) at the top right. Now click anywhere in the white area.</li>
        <li>This creates a new carbon atom. When you hover your mouse over the carbon atom, it will be highlighted in green.</li>
    </ul>
    </li>

    <li>
    The <i>left sidebar</i> has a list of common chemical bonds. 
    <ul>
        <li>Click on the line (single bond) at the top left.</li>
        <li>Now hover on the carbon atom and click <i>three times</i>. This creates three single bonds joined to the carbon atom.</li>
    </ul>
    </li>
    
    <li>
    Lastly, click on the "Cl" (chlorine) atom on the right, and then click on the ends of the bonds to change each atom on the end to a Cl atom.
    <ul>
        <li>Notice we don't need to add hydrogen atoms! Molview assumes that any un-bonded outer shell electrons will find a hydrogen, and adds them in automatically.</li>
    </ul>
    </li>
</ol>
''')


@solara.component
def Step2():
    with solara.Details(summary=solara.Markdown('### Step 2: Running quantum geometry calculations'), expand=False):
        with solara.lab.Tabs():
            with solara.lab.Tab('Instructions: Click the buttons!'):
                solara.Markdown(
'''
1. Underneath this box, click the "Run Quantum Optimization" button. This will take a little time, and then print out when it's done.
2. Once the calculation is done, click the "Show Minimized State"  button to see the molecule in 3D!
''')
            with solara.lab.Tab('Information: Energy Minimization and Molecular Vibration Searches'):
                solara.Markdown(
'''
This notebook runs through two calculations. The first is an _energy minimization_ (also called geometry optimization),
where the underlying program adjusts the coordinates of individual atoms, and then calculates the total energy the
molecule would have in that state. These calculations are repeated until a state is found at which the molecule has
minimum energy - like a ball rolling down to the bottom of a bowl, this will be the most stable state for the molecule.

The next calculation is a _vibrational analysis_. The program adjusts the atoms' coordinates again - but instead of
finding the minimum energy, it records how the energy changes with the atoms' movement, and separates this into
molecular vibrations - like a ball now rattling around near the bottom of a bowl. A molecule's vibrations can be
measured experimentally by seeing which light frequencies are best absorbed by the molecule, and they're also
important for simulating the molecule later on!
''')


@solara.component
def Step3():
    with solara.Details(summary=solara.Markdown('### Step 3: Calculating molecular vibrations'), expand=False):
        with solara.lab.Tabs():
            with solara.lab.Tab('Instructions: Click the buttons!'):
                solara.Markdown(
'''
<ol>
    <li><i>This next step may take <b>up to 20 minutes</b> for molecules with <b>more than twelve atoms</b></i>:
        <ul><li>click the "Find Molecular Vibrations" button below.</li></ul>
    </li>
    <li>This will start the quantum calculations. While you wait for results, you can do the next section ("Measure angles and distances in Molview").</li>
    <li>
    Now click the "Show Vibrations" button. You will see a list of vibration frequencies. Selecting a frequency will show which atoms in the molecule vibrate at that frequency.
    <ul>
        <li>Is there a pattern to which atoms vibrate at lower frequencies (smaller numbers), and which atoms vibrate at higher frequencies?</li>
        <li>The "THz" unit is 1 _terahertz_, or a trillion vibrations per second!</li>
    </ul>
    </li>
</ol>
''')
            with solara.lab.Tab('Information: Vibrations and Spectroscopy'):
                solara.Markdown(
'''
In gases, molecules only vibrate at certain frequencies (as your quantum calculation shows).
This means they also only absorb light at certain frequencies, or wavelengths.
The frequencies you have calculated will fall between 1 and 400 THz, which falls under _infrared radiation_.
Infrared radiation can't be seen by human eyes, but we can feel it as heat (such as from a barbecue grill!).

The Earth's atmosphere stays warm (and is getting warmer) because gases in the atmosphere absorb infrared radiation.
By measuring which wavelengths different molecules absorb, scientists can also get information about how long bonds
are in molecules, giving more clues about their structure. Finally, knowing how molecules vibrate is important for
simulating them - like you're going to do later today!
''')


@solara.component
def Step4():
    with solara.Details(summary=solara.Markdown('### Step 4: Measure angles and distances in Molview'), expand=False):
        with solara.lab.Tabs():
            with solara.lab.Tab('Instructions: Molview Measurements'):
                solara.Markdown(
'''
<ol>
    <li>If you'd like to simulate your own molecule, you can get a SMILES code from Molview using the process in Step 1.</li>

    <li>
    While you're waiting for the quantum calculations to run, you can also use Molview to measure approximate bond lengths and angles:
    <ol>
        <li>Look for the "2D to 3D" button on the top bar. Click it to create an approximate 3D representation of your molecule!</li>
        <li>Now look for the "Jmol" menu on the top bar. When you open the menu, you will see three options at the bottom under "Measurement": Distance, Angle and Torsion.</li>
        <li>Measure different bond lengths and angles in your molecule (see the "Bond Length" and "Angle" tabs for detailed instructions). Are each of the bond lengths the same, or different? Are each of the angles the same, or different? How do the measurements compare to your quantum results? Try to make sense of what you see.</li>
    </ol
    </li>

    <li>
    You can also look at the polarity of your molecule!
    <ul>
        <li>In the "Jmol" menu, click on "MEP Surface Lucent". This will show a surface around your molecule representing the electric potential energy (using Coulomb's law!). Bluer areas are more positive, while redder areas are more negative.</li>
    </ul>
    </li>
</ol>
''')
            with solara.lab.Tab('Bond Lengths'):
                solara.Markdown(
'''
1. In the "Jmol" menu, click the "Distance" option.
2. Now click on one of the atoms in the _3D_ view of the molecule. When you move the mouse away, you should now see a dotted line following your mouse position.
3. Click on another atom. Now Molview will show the distance between the two chosen atoms (which is the bond length, if those atoms share a bond). The letters "nm" stand for _nanometer_, or one billionth of a meter.
4. To remove all the numbers being shown, use the "Clear" button in the "Jmol" menu.
''')
            with solara.lab.Tab('Bond Angles'):
                solara.Markdown(
'''
1. In the "Jmol" menu, click the "Angle" option.
2. Now click on one of the atoms in the _3D_ view of the molecule. When you move the mouse away, you should now see a dotted line following your mouse position.
3. Click on another two atoms. Now Molview will show the angle formed between the three chosen atoms, in degrees.
4. To remove all the numbers being shown, click again on the "2D to 3D" button you used earlier.
''')
            with solara.lab.Tab('Information: Bonds, Angles and Tortions'):
                solara.Markdown(
'''
A molecule's geometry can be described in terms of how long bonds are between atoms, and what angles
those bonds make to each other. (A _torsion_ is the angle formed between _four_ atoms, which changes
when one pair twists against the second pair.) This lets computer programs predict how molecules will
move by treating them as little balls and sticks!

The atoms between bonds are too small to see directly -- so how do scientists know what shapes molecules have?
One of the most important ways is through _X-ray diffraction_. In a solid crystal, molecules are arranged in
a regular order, and strong X-ray beams can reveal the spacings between different atoms.

A famous example of this was when Rosalind Franklin produced the first X-ray diffraction pictures of DNA
crystals in the 1950s. Her data was necessary for James Watson, Francis Crick and Maurice Wilkins to piece
together the exact arrangement of atoms in the DNA double helix, for which they won the Nobel Prize in 1962.
''')
