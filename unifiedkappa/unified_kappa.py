import itertools
from functools import partial
from pathlib import Path

import numpy as np
import phonopy
from phonopy import Phonopy
from phonopy.structure.symmetry import Symmetry, _symmetrize_2nd_rank_tensor

from shengbte_analysis import ShengBTEAnalyzer

"""
author: @yixia
author: @dgaines2
A python class to compute kL_min

The calculation of kL_min relies on modified versions of:
    - api_phonopy.py -> from phonopy import Phonopy
    - mesh.py -> from phonopy.phonon.mesh import Mesh
    - group_velocity.py -> from phonopy.phonon.group_velocity import GroupVelocity
for Phonopy version 2.17.1
"""


class LibraryModificationRequired(Exception):
    """Raised when the required library modifications have not been made."""
    pass


class UnifiedkappaManager:
    def __init__(
        self,
        phonon,
        mesh=25.0,
        shengbte_dir=None,
        n_histogram_bins=50,
        save_histogram=False,
    ):
        """
        Args:
            phonon (phonopy.Phonopy)
            mesh (float | np.array([3,], dtype=int)): q-point mesh or mesh density
            shengbte_dir (str | Path): directory for ShengBTE outputs
            n_histogram_bins (int): if save_histogram is True, the number of
                bins for diagonal and off diagonal components of unifiedkappa
            save_histogram (bool): if True, write files of binned diagonal and
                off diagonal components of unifiedkappa
        """
        self.phonon = phonon
        self.mesh = (
            np.asarray(mesh, dtype=int) if isinstance(mesh, np.ndarray) else mesh
        )
        self.shengbte_dir = shengbte_dir
        self.n_histogram_bins = n_histogram_bins
        self.save_histogram = save_histogram

    def get_mesh_dict(self):
        try:
            self.phonon.run_mesh(
                self.mesh,
                with_eigenvectors=False,
                is_gamma_center=True,
                with_full_group_velocities=True,
                is_time_reversal=False,
                is_mesh_symmetry=False,
            )
        except TypeError as e:
            if "full_group_velocities" in str(e):
                raise LibraryModificationRequired(
                    "Phonopy must be modified to calculate off-diagonal group "
                    "velocities. Be sure to make modifications to Phonopy 2.17.1. "
                    "See more details in the README here: "
                    "https://github.com/yimavxia/Minikappa/tree/main/scripts"
                ) from e
            else:
                raise
        mesh_dict = self.phonon.get_mesh_dict()
        return mesh_dict

    @staticmethod
    def get_maximum_scattering_rates(freqs, tau_factor):
        """
        Args:
            freqs (np.array[nqpt, nband]): frequencies of each phonon mode in
                units of 2*pi*THz
            tau_factor (float)
                Note: tau_factor=2 corresponds to the assumption from our paper
        Returns:
            Gamma (np.array(nqpt, nband]): scattering rate for each phonon mode
        """
        nqpt, nband = freqs.shape
        Gamma = np.ones((nqpt, nband)) * 1e10
        for iq, i in itertools.product(range(nqpt), range(nband)):
            omega = freqs[iq, i]
            if omega > 0:
                Gamma[iq, i] = omega / 2 / np.pi * tau_factor
        return Gamma

    @staticmethod
    def get_planckian_scattering_rates(freqs, temperature):
        nqpt, nband = freqs.shape
        Gamma = np.ones((nqpt, nband)) * 1e10
        for iq, i in itertools.product(range(nqpt), range(nband)):
            omega = freqs[iq, i]
            if omega > 0:
                Gamma[iq, i] = 0.13092 * temperature
        return Gamma

    def calculate_unified_kappa(
        self,
        freqs,
        Gamma,
        gvfull,
        temperature=300.0,
        freqcf=0.1,
        symprec=1e-03,
        filename_prefix=None,
    ):
        """
        Args:
            freqs (np.array[nqpt, nband], dtype=float): phonon frequencies in 2*pi*THz
            Gamma (np.array[nqpt, nband], dtype=float): phonon scattering rates
                in units of ps^(-1)
            gvfull (np.array[nqpt, nband, nband, 3], dtype=complex): full diagonal and
                off-diagonal group velocities in km/s
            temperature (float): temperature in Kelvin
            freqcf (float): cutoff frequency. Any frequency below this value will not
                contribute to the thermal conductivity
            symprec (float): if not None, symprec for determining the symmetry
                of the crystal structure. This will be used to symmetrize the final
                tensors to be invariant with respect to the symmetry operations of
                the crystal structure
            filename_prefix (str): beginning of filename for saving outputs
        """
        # Units
        hbar = 1.054571726470000e-022
        kB = 1.380648813000000e-023

        volpc = self.phonon.primitive.volume / 1000.0  # Angs^3 to nm^3
        nqpt, nband = freqs.shape

        delta_freq = np.max(freqs + 1e-01) / self.n_histogram_bins
        histogram_kappa_d = np.zeros(
            (self.n_histogram_bins, self.n_histogram_bins, 3, 3)
        )
        histogram_kappa_od = np.zeros(
            (self.n_histogram_bins, self.n_histogram_bins, 3, 3)
        )

        kappaband = np.zeros((nband, nband, 3, 3), dtype=np.complex128, order="C")
        for iq, i, j, k, kp in itertools.product(
            range(nqpt),
            range(nband),
            range(nband),
            range(3),
            range(3),
        ):
            omega1 = freqs[iq, i]
            omega2 = freqs[iq, j]
            if omega1 <= freqcf or omega2 <= freqcf:
                continue
            Gamma1 = Gamma[iq, i]
            Gamma2 = Gamma[iq, j]
            fBE1 = 1.0 / (np.exp(hbar * omega1 / kB / temperature) - 1.0)
            fBE2 = 1.0 / (np.exp(hbar * omega2 / kB / temperature) - 1.0)
            tmpv = (gvfull[iq, i, j, k] * gvfull[iq, j, i, kp]).real
            kappaband_tmp = (omega1+omega2)/2 * \
                (fBE1*(fBE1+1)*omega1+fBE2*(fBE2+1)*omega2) * tmpv \
                / (4*(omega1-omega2)**2+(Gamma1+Gamma2)**2) \
                * (Gamma1+Gamma2)
            kappaband[i, j, k, kp] += kappaband_tmp

            idx_freq1 = int(omega1 // delta_freq)
            idx_freq2 = int(omega2 // delta_freq)
            if i == j:
                histogram_kappa_d[idx_freq1, idx_freq2, k, kp] += kappaband_tmp
            else:
                histogram_kappa_od[idx_freq1, idx_freq2, k, kp] += kappaband_tmp

        # conversion
        unit_factor = 1e21 * hbar**2 / (kB * temperature**2 * volpc * nqpt)
        kappaband *= unit_factor
        histogram_kappa_d *= unit_factor
        histogram_kappa_od *= unit_factor
        if self.save_histogram:
            if filename_prefix is None:
                filename_prefix = f"unifiedkappa-{int(temperature)}"
            for direction, index in zip(["xx", "yy", "zz"], [0, 1, 2]):
                np.savetxt(
                    f"{filename_prefix}-d_{direction}.txt",
                    histogram_kappa_d[:, :, index, index],
                )
                np.savetxt(
                    f"{filename_prefix}-od_{direction}.txt",
                    histogram_kappa_od[:, :, index, index],
                )

        kappaD = np.zeros((3, 3), dtype=np.complex128, order="C")
        kappaOD = np.zeros((3, 3), dtype=np.complex128, order="C")
        kappaF = np.zeros((3, 3), dtype=np.complex128, order="C")
        for i, j in itertools.product(range(nband), range(nband)):
            kappaF += kappaband[i, j]
            if i == j:
                kappaD += kappaband[i, j]
            else:
                kappaOD += kappaband[i, j]
        kappaD = kappaD.real
        kappaOD = kappaOD.real
        kappaF = kappaF.real
        if symprec is not None:
            ph_symm = Symmetry(
                self.phonon.primitive,
                symprec=symprec,
                is_symmetry=True,
            )
            symmetrize_2nd_rank_tensor = partial(
                _symmetrize_2nd_rank_tensor,
                symmetry_operations=ph_symm.pointgroup_operations,
                lattice=self.phonon.primitive.cell,
            )
            kappaD = symmetrize_2nd_rank_tensor(kappaD)
            kappaOD = symmetrize_2nd_rank_tensor(kappaOD)
            kappaF = symmetrize_2nd_rank_tensor(kappaF)
        return kappaD, kappaOD, kappaF

    def run_minikappa(
        self,
        temperatures=[300.0, 600.0, 900.0],
        tau_factors=[2.0],
        output_dir=None,
        verbose=True,
    ):
        def vprint(message, verbose=True):
            if verbose:
                print(message)

        # Mesh
        vprint(f"Running phonon mesh... mesh={self.mesh}", verbose)
        mesh_dict = self.get_mesh_dict()
        freqs = mesh_dict["frequencies"] * 2 * np.pi  # THz -> 2*pi*THz
        gvfull = mesh_dict["group_velocities_full"] / 10.0  # Angs*THz -> nm*THz == km/s

        # kappa
        results = {}
        for temperature in temperatures:
            results[temperature] = {}
            for tau_factor in tau_factors:
                results[temperature][tau_factor] = {}
                vprint(
                    f"Calculating minikappa... T={temperature}K, tau={tau_factor}",
                    verbose,
                )
                Gamma = self.get_maximum_scattering_rates(freqs, tau_factor=tau_factor)
                if output_dir is None:
                    output_dir = Path(".")
                filename_prefix = str(
                    output_dir / f"minikappa-{int(temperature)}-{tau_factor}"
                )
                kappaD, kappaOD, kappaF = self.calculate_unified_kappa(
                    freqs,
                    Gamma,
                    gvfull,
                    temperature=float(temperature),
                    filename_prefix=filename_prefix,
                )
                results[temperature][tau_factor]["D"] = kappaD
                results[temperature][tau_factor]["OD"] = kappaOD
                results[temperature][tau_factor]["F"] = kappaF

                output_filename = f"{filename_prefix}.dat"
                vprint(f"Writing to {output_filename}", verbose)
                with open(output_filename, "w+") as fw:
                    for kappa_matrix in [kappaD, kappaOD, kappaF]:
                        kappa_matrix = np.round(
                            kappa_matrix.flatten(),
                            decimals=8,
                        )
                        fw.write(
                            "".join([f"{num:>14.8f}" for num in kappa_matrix]) + "\n"
                        )
                kappaD_scalar = np.mean(np.diag(kappaD))
                kappaOD_scalar = np.mean(np.diag(kappaOD))
                kappaF_scalar = np.mean(np.diag(kappaF))

                vprint(
                    "Diagonal part of minimum thermal conductivity: "
                    + f"{kappaD_scalar:.3f} W/m/K",
                    verbose,
                )
                vprint(
                    "Off-diagonal part of minimum thermal conductivity: "
                    + f"{kappaOD_scalar:.3f} W/m/K",
                    verbose,
                )
                vprint(
                    f"Total minimum thermal conductivity: {kappaF_scalar:.3f} W/m/K",
                    verbose,
                )
                vprint("", verbose)
        return results

    def run_unified_kappa(self, verbose=True):
        def vprint(message, verbose=True):
            if verbose:
                print(message)

        if self.shengbte_dir is None:
            raise RuntimeError("shengbte_dir must be set to run unified kappa")

        # Mesh
        vprint(f"Running phonon mesh... mesh={self.mesh}", verbose)
        mesh_dict = self.get_mesh_dict()
        freqs = mesh_dict["frequencies"] * 2 * np.pi  # THz -> 2*pi*THz
        gvfull = mesh_dict["group_velocities_full"] / 10.0  # Angs*THz -> nm*THz == km/s

        # kappa
        results = {}
        sbte = ShengBTEAnalyzer(self.shengbte_dir, scattering_rate_cutoff=0.0)
        for temperature in sbte.temperatures:
            results[temperature] = {}
            vprint(
                f"Calculating unified kappa... T={temperature}K",
                verbose,
            )
            Gamma = sbte.rates_rta_fbz[f"{temperature}"]
            filename_prefix = str(
                self.shengbte_dir / f"T{temperature}K" / "unifiedkappa"
            )
            kappaD, kappaOD, kappaF = self.calculate_unified_kappa(
                freqs,
                Gamma,
                gvfull,
                temperature=temperature,
                filename_prefix=filename_prefix,
            )
            results[temperature]["D"] = kappaD
            results[temperature]["OD"] = kappaOD
            results[temperature]["F"] = kappaF

            output_filename = f"{filename_prefix}.dat"
            vprint(f"Writing to {output_filename}", verbose)
            with open(output_filename, "w+") as fw:
                for kappa_matrix in [kappaD, kappaOD, kappaF]:
                    kappa_matrix = np.round(
                        kappa_matrix.flatten(),
                        decimals=8,
                    )
                    fw.write("".join([f"{num:>15.8f}" for num in kappa_matrix]) + "\n")
            kappaD_scalar = np.mean(np.diag(kappaD))
            kappaOD_scalar = np.mean(np.diag(kappaOD))
            kappaF_scalar = np.mean(np.diag(kappaF))

            vprint(
                "Diagonal part of thermal conductivity: "
                + f"{kappaD_scalar:.3f} W/m/K",
                verbose,
            )
            vprint(
                "Off-diagonal part of thermal conductivity: "
                + f"{kappaOD_scalar:.3f} W/m/K",
                verbose,
            )
            vprint(
                f"Total thermal conductivity: {kappaF_scalar:.3f} W/m/K",
                verbose,
            )
            vprint("", verbose)
        return results

    @classmethod
    def from_parameters(
        cls,
        poscar_path,
        supercell_matrix,
        primitive_matrix,
        force_constants_filename="FORCE_CONSTANTS",
        **kwargs,
    ):
        """
        Construct UnifiedkappaManager from a set of parameters

        Args:
            poscar_path (str | Path): path to POSCAR file
            supercell_matrix (3x3 array[int]): supercell matrix
            primitive_matrix (3x3 array[float]): primitive matrix
            force_constants_filename (str | Path): path to harmonic force
                constants file
        """
        phonon = phonopy.load(
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            unitcell_filename=str(poscar_path),
            force_constants_filename=str(force_constants_filename),
            is_symmetry=False,
        )
        return cls(phonon, **kwargs)

    @classmethod
    def from_control(
        cls,
        shengbte_dir,
        poscar_filename="POSCAR",
        force_constants_filename="FORCE_CONSTANTS_2ND",
        **kwargs,
    ):
        """
        Construct UnifiedkappaManager from a ShengBTE CONTROL file in
        shengbte_dir. This method is preferred as the mesh is matched with
        ngrid from CONTROL.

        Args:
            shengbte_dir (str | Path)
            poscar_filename (str): path to POSCAR filename in shengbte_dir
            force_constants_filename (str): harmonic force constants filename
                in shengbte_dir
        """
        shengbte_dir = Path(shengbte_dir)
        sbte = ShengBTEAnalyzer(shengbte_dir)
        poscar_path = str(shengbte_dir / poscar_filename)
        force_constants_path = str(shengbte_dir / force_constants_filename)

        phonon = phonopy.load(
            supercell_matrix=sbte.scell_matrix,
            primitive_matrix=np.eye(3),
            unitcell_filename=poscar_path,
            force_constants_filename=force_constants_path,
            is_symmetry=False,
        )
        return cls(phonon, mesh=sbte.ngrid, shengbte_dir=shengbte_dir, **kwargs)


def read_minikappa_file(fpath, verbose=False):
    minikappa_output = np.loadtxt(fpath)
    kappaD, kappaOD, kappaF = minikappa_output.reshape(3, 3, 3)
    kappaD_scalar = np.mean(np.diag(kappaD))
    kappaOD_scalar = np.mean(np.diag(kappaOD))
    kappaF_scalar = np.mean(np.diag(kappaF))
    if verbose:
        print(f"Diagonal part of thermal conductivity: {kappaD_scalar:.3f} W/m/K")
        print(f"Off-diagonal part of thermal conductivity: {kappaOD_scalar:.3f} W/m/K")
        print(f"Total thermal conductivity: {kappaF_scalar:.3f} W/m/K")
    return kappaD, kappaOD, kappaF


if __name__ == "__main__":
    """
    Here's an example of using from_control to initialize UnifiedkappaManager
    Run this in ./shengbte_example
    """
    shengbte_dir = Path(".")
    poscar_filename = "POSCAR-prim"
    unifiedkappa_manager = UnifiedkappaManager.from_control(
        shengbte_dir=shengbte_dir,
        poscar_filename=poscar_filename,
    )
    unifiedkappa_results = unifiedkappa_manager.run_unified_kappa()
    minikappa_results = unifiedkappa_manager.run_minikappa(
        temperatures=[300.0],
        tau_factors=[2.0],
        verbose=True,
    )

    """
    Here's an example of using from_parameters to initialize UnifiedkappaManager
    - This is useful in the case where you only want to get minikappa from
      harmonic phonon outputs
    Run this in ./phonopy_example
    """
    # unifiedkappa_manager = UnifiedkappaManager.from_parameters(
    #     poscar_path=poscar_filename,
    #     supercell_matrix=np.eye(3) * 4,
    #     primitive_matrix=np.eye(3),
    #     force_constants_filename="FORCE_CONSTANTS",
    #     mesh=[12, 12, 12],
    # )
    # minikappa_results = unifiedkappa_manager.run_minikappa(
    #     temperatures=[300.0, 600.0, 900],
    #     tau_factors=[2.0],
    #     verbose=True,
    # )
