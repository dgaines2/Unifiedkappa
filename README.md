# Unifiedkappa
Scripts for computing the diagonal and off-diagonal contributions to lattice thermal conductivity, based on the inputs and outputs from ShengBTE. We provide revised phonopy files that can produce off-diagonal group velocity tensors with degeneracy properly treated.
# How to use the scripts
> [!IMPORTANT] 
> Replace the original phonopy files with the modified ones found in `phonopy_files` 
>  - Be sure to use phonopy version 2.17.1
>  - `api_phonopy.py` &rarr; `from phonopy import Phonopy`
>  - `mesh.py` &rarr; `from phonopy.phonon.mesh import Mesh`
>  - `group_velocity.py` &rarr; `from phonopy.phonon.group_velocity import GroupVelocity`

> [!NOTE]
> Don't worry, we only extend the functionality and all original functionality is maintained!

* Go to the folder `unifiedkappa/example` and run `python ../unified_kappa.py`
  - This folder contains the inputs and outputs typically found in ShengBTE calculations
* Note that in order to read scattering rates produced by ShengBTE, you must explicitly specify `onlyharmonic=.FALSE.` in the `CONTROL` file
