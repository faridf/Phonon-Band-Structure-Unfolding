# Phonon Band Structure Unfolding

This Python script implements phonon band structure unfolding based on the method described in P. B. Allen et al. [Phys. Rev. B 87, 085322 (2013)](https://doi.org/10.1103/PhysRevB.87.085322). The script unfolds phonon bands from a supercell calculation back to the primitive cell Brillouin zone, which is particularly useful for analyzing systems with defects, disorders, or complex structures.

## Features

- Phonon band unfolding using P. B. Allen's method
- Handles arbitrary supercell sizes
- Publication-quality plotting with customizable options
- Support for BCC structures (easily adaptable for other structures)
- Weight-based visualization showing band character
- Automatic high-symmetry path generation

## Dependencies

- Python 3.6+
- phonopy
- numpy
- matplotlib
- ASE (Atomic Simulation Environment)

Install the required packages using pip:
```bash
pip install phonopy numpy matplotlib ase
```

## Usage

### Required Input Files

1. `POSCAR`: VASP format structure file
2. `FORCE_CONSTANTS`: Force constants file generated by phonopy

### Running the Script

1. Place the required input files in your working directory
2. Run the script:
```bash
python phonon_unfolding.py
```

The script will generate an unfolded phonon band structure plot saved as `phonon_bands_unfolded.png`.

### Customizing the Calculation

You can modify the main parameters in the script:

```python
ax = phonopy_unfold(
    sc_mat=np.eye(3),              # Primitive cell to supercell transformation matrix
    unfold_sc_mat=np.diag([6,6,6]), # Supercell matrix for unfolding
    force_constants='FORCE_CONSTANTS',
    sposcar='POSCAR',
    qpts=kpts,                     # q-points for the band path
    qnames=names,                  # Names of high-symmetry points
    xqpts=x,                       # x-coordinates for plotting
    Xqpts=X)                       # x-coordinates of high-symmetry points
```

## Technical Details

### Unfolding Method

The unfolding procedure follows these steps:

1. Maps atomic positions between primitive and supercell using translation vectors
2. Calculates spectral weights using the formula:
   ```
   W = ∑(1/N) ⟨ψ|T(r)exp(-iKr)|ψ⟩
   ```
   where:
   - ψ is the eigenvector
   - T(r) is the translation operator
   - K is the wave vector
   - N is the number of translations

3. Projects supercell eigenvectors onto primitive cell basis
4. Visualizes bands with opacity proportional to their spectral weight

### Band Structure Plot

The output plot includes:
- Frequencies in THz
- High-symmetry path with labeled points
- Band weights shown through color intensity
- Publication-ready formatting and styling

## Output

The script generates:
- `phonon_bands_unfolded.png`: The unfolded phonon band structure plot
- Frequency range starting from 0 THz
- Weights visualized through color intensity (darker = higher weight)

## Example Output
![phonon_bands_unfolded](https://github.com/user-attachments/assets/9ce8c3fc-d9bd-470a-8d7f-f4a741fd81a1)

## Customization

You can customize the plot appearance by modifying the `plot_band_weight` function:
- Change figure dimensions
- Modify colors and line styles
- Adjust font sizes and label formatting
- Change grid properties
- Customize axis ranges and tick marks

## Citation

If you use this script in your research, please cite:
1. P. B. Allen et al., Phys. Rev. B 87, 085322 (2013)
2. Phonopy: https://phonopy.github.io/phonopy/
3. This repository


## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.


## Acknowledgments

- Phonopy developers
- ASE developers
- Original authors of the unfolding method
