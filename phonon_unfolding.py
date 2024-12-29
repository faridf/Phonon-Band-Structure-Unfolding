#!/usr/bin/env python

from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_CONSTANTS
from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter
from ase.build import make_supercell
from ase.atoms import Atoms

def read_phonopy(sposcar='SPOSCAR', sc_mat=np.eye(3), force_constants='FORCE_CONSTANTS'):
    """Read phonopy data with proper primitive cell setup"""
    atoms = read_vasp(sposcar)
    primitive_matrix = inv(sc_mat)
    
    bulk = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.get_cell())

    phonon = Phonopy(
        bulk,
        supercell_matrix=np.eye(3),
        primitive_matrix=primitive_matrix)
    
    fc = parse_FORCE_CONSTANTS(force_constants)
    phonon.force_constants = fc
    
    return phonon

class PhononUnfolder:
    """Phonon unfolding class following Allen's method"""
    def __init__(self, atoms, supercell_matrix, eigenvectors, qpoints, tol_r=0.04, ndim=3, labels=None, phase=False):
        self._atoms = atoms
        self._scmat = supercell_matrix
        self._evecs = eigenvectors
        self._qpts = qpoints
        self._tol_r = tol_r
        self._ndim = ndim
        self._labels = labels
        self._phase = phase
        
        self._trans_rs = None
        self._trans_indices = None
        self._make_translate_maps()
    
    def _make_translate_maps(self):
        """Create translation vectors and mapping between atoms"""
        # Get translation vectors
        a1 = Atoms(symbols='H', positions=[(0, 0, 0)], cell=[1, 1, 1])
        sc = make_supercell(a1, self._scmat)
        rs = sc.get_scaled_positions()
        
        # Get positions and prepare indices array
        positions = np.array(self._atoms.get_scaled_positions())
        indices = np.zeros([len(rs), len(positions) * self._ndim], dtype='int32')
        
        # Map atoms
        for i, ri in enumerate(rs):
            Tpositions = positions + np.array(ri)
            for i_atom, pos in enumerate(positions):
                for j_atom, Tpos in enumerate(Tpositions):
                    dpos = Tpos - pos
                    if np.all(np.abs(dpos - np.round(dpos)) < self._tol_r):
                        indices[i, j_atom * self._ndim:(j_atom + 1) * self._ndim] = \
                            np.arange(i_atom * self._ndim, (i_atom + 1) * self._ndim)
        
        self._trans_rs = rs
        self._trans_indices = indices
    
    def get_weight(self, evec, qpt, G=np.array([0,0,0])):
        """Calculate spectral weight for a mode"""
        weight = 0j
        N = len(self._trans_rs)
        
        for r_i, ind in zip(self._trans_rs, self._trans_indices):
            if self._phase:
                weight += np.vdot(evec, evec[ind])*np.exp(-1j *2 * np.pi * np.dot(qpt+G,r_i)) /N
            else:
                weight += np.vdot(evec, evec[ind]) / N*np.exp(-1j *2 * np.pi * np.dot(G,r_i))
        
        return weight.real
    
    def get_weights(self):
        """Get weights for all modes"""
        nqpts, nfreqs = self._evecs.shape[0], self._evecs.shape[1]
        weights = np.zeros([nqpts, nfreqs])
        
        for iqpt in range(nqpts):
            for ifreq in range(nfreqs):
                weights[iqpt, ifreq] = self.get_weight(
                    self._evecs[iqpt, :, ifreq],
                    self._qpts[iqpt])
        
        return weights

def plot_band_weight(kslist, ekslist, wkslist=None, efermi=0, yrange=None,
                    output=None, style='alpha', color='blue', axis=None,
                    width=2, xticks=None, title=None):
    """Plot band structure with weights"""
    # Convert frequencies from cm^-1 to THz
    ekslist = np.array(ekslist) / 33.356  # Convert from cm^-1 back to THz
    
    if axis is None:
        fig, a = plt.subplots(figsize=(8, 6))
        plt.tight_layout(pad=2.19)
        plt.axis('tight')
        plt.gcf().subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.12)
    else:
        a = axis
    if title is not None:
        a.set_title(title, fontsize=12, pad=10)
    
    xmax = max(kslist[0])
    if yrange is None:
        ymax = np.array(ekslist).flatten().max() * 1.1
        yrange = (0, ymax)  # Force y-axis to start from 0
    
    if wkslist is not None:
        for i in range(len(kslist)):
            x = kslist[i]
            y = ekslist[i]
            lwidths = np.array(wkslist[i]) * width
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            if style == 'width':
                lc = LineCollection(segments, linewidths=lwidths, colors=color)
            elif style == 'alpha':
                lc = LineCollection(
                    segments,
                    linewidths=[1.5] * len(x),  # Slightly thinner lines
                    colors=[
                        colorConverter.to_rgba(
                            'royalblue', alpha=np.abs(lwidth / (width + 0.001)))  # Changed color
                        for lwidth in lwidths])
            a.add_collection(lc)
    
    # Customize plot
    plt.ylabel('Frequency (THz)', fontsize=12, labelpad=10)
    plt.xlabel('Wave Vector', fontsize=12, labelpad=10)
    
    if axis is None:
        # Set axis limits
        a.set_xlim(0, xmax)
        a.set_ylim(yrange)
        
        # Set ticks and grid
        if xticks is not None:
            plt.xticks(xticks[1], xticks[0], fontsize=11)
        for x in xticks[1]:
            plt.axvline(x, color='gray', linewidth=0.5, alpha=0.5)
        
        # Add horizontal gridlines
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Customize tick parameters
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tick_params(axis='x', which='major', pad=8)
        
        # Add minor ticks on y-axis
        a.yaxis.set_minor_locator(plt.AutoLocator())
        
        if efermi is not None:
            plt.axhline(linestyle='--', color='black', alpha=0.5)
    
    return a

def unf(phonon, sc_mat, qpoints, knames=None, x=None, xpts=None):
    """Core unfolding function"""
    # Get primitive cell
    prim = phonon.get_primitive()
    prim = Atoms(symbols=prim.get_chemical_symbols(),
                 cell=prim.get_cell(),
                 positions=prim.get_positions())
    
    # Transform q-points
    sc_qpoints = np.array([np.dot(q, sc_mat) for q in qpoints])
    
    # Get phonon data
    phonon.set_qpoints_phonon(sc_qpoints, is_eigenvectors=True)
    freqs, eigvecs = phonon.get_qpoints_phonon()
    
    # Do unfolding
    uf = PhononUnfolder(atoms=prim,
                        supercell_matrix=sc_mat,
                        eigenvectors=eigvecs,
                        qpoints=sc_qpoints,
                        phase=False)
    weights = uf.get_weights()
    
    # Plot bands (Note: no frequency conversion needed as we'll handle it in plotting)
    ax = plot_band_weight([list(x)]*freqs.shape[1],
                         freqs.T*33.356,  # Still convert to cm^-1 for internal consistency
                         weights[:,:].T*0.99+0.001,
                         xticks=[knames,xpts],
                         style='alpha',
                         title='Unfolded Phonon Band Structure')
    return ax

def phonopy_unfold(sc_mat=np.diag([1,1,1]),
                  unfold_sc_mat=np.diag([3,3,3]),
                  force_constants='FORCE_CONSTANTS',
                  sposcar='SPOSCAR',
                  qpts=None,
                  qnames=None,
                  xqpts=None,
                  Xqpts=None):
    """Main unfolding function"""
    phonon = read_phonopy(sc_mat=sc_mat,
                         force_constants=force_constants,
                         sposcar=sposcar)
    ax = unf(phonon,
            sc_mat=unfold_sc_mat,
            qpoints=qpts,
            knames=qnames,
            x=xqpts,
            xpts=Xqpts)
    return ax

def main():
    from ase.dft.kpoints import get_special_points, bandpath
    
    # Read structure
    print("Reading structure...")
    atoms = read_vasp("POSCAR")
    print(f"Read structure with {len(atoms)} atoms")
    
    # Get band path
    points = get_special_points('bcc', atoms.cell)
    path = [points[k] for k in 'GHPGN']
    kpts, x, X = bandpath(path, atoms.cell, npoints=400)
    names = ['$\Gamma$', 'H', 'P', '$\Gamma$', 'N']
    
    print("\nCalculating unfolded bands...")
    ax = phonopy_unfold(
        sc_mat=np.eye(3),
        unfold_sc_mat=np.diag([6,6,6]),
        force_constants='FORCE_CONSTANTS',
        sposcar='POSCAR',
        qpts=kpts,
        qnames=names,
        xqpts=x,
        Xqpts=X)
    
    plt.savefig('phonon_bands_unfolded.png', dpi=300, bbox_inches='tight')
    print("\nBand structure saved as 'phonon_bands_unfolded.png'")

if __name__ == "__main__":
    main()