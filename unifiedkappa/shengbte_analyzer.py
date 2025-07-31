from collections import defaultdict
from pathlib import Path

import f90nml
import numpy as np
import phonopy

"""
author: @dgaines2
"""


class ShengbteAnalyzer:
    """
    Simple class to extract information from ShengBTE/FourPhonon inputs and outputs
    """

    def __init__(
        self,
        workdir,
        scattering_rate_cutoff=0.0,
        kappa_precision=5,
    ):
        """
        Args:
            workdir (str | Path): path to the directory of the calculation
            scattering_rate_cutoff (float): any scattering rates at or below this value
                will be replaced with a very large number (=1e10) in order to prevent
                unphysically long phonon lifetimes. Set None to get the raw data.
            kappa_precision (int): precision to round the thermal conductivity tensor
        """
        self.workdir = Path(workdir).resolve()
        self.scattering_rate_cutoff = scattering_rate_cutoff
        self.kappa_precision = kappa_precision

        self._check_complete()
        self._read_control()
        self._read_frequencies()
        self._read_dos()
        self._read_qpoints()
        self._read_gruneisen()
        self._read_group_velocities()
        self._read_phase_space()
        self._read_scattering_rates()
        self._read_kappa()
        self._read_unified_kappa()

    def _check_complete(self):
        bte_out_path = self.workdir / "BTE.out"
        if not bte_out_path.exists():
            raise FileNotFoundError(f"No BTE.out file in {self.workdir}... exiting")
        with open(bte_out_path) as fr:
            bte_out = [line.strip() for line in fr.readlines()]
        complete = False
        # goodlines = ["normal exit", "onlyharmonic=.true., stopping here"]
        goodlines = ["normal exit"]
        for line in bte_out:
            if any(goodline in line for goodline in goodlines):
                complete = True
        if not complete:
            raise RuntimeError(f"BTE calculation failed in {self.workdir}... exiting")

    def _read_control(self):
        control_path = self.workdir / "CONTROL"
        raw_control_dict = f90nml.read(control_path).todict()
        control = {}
        control.update(raw_control_dict["allocations"])
        control.update(raw_control_dict["crystal"])
        control.update(raw_control_dict["parameters"])
        control.update(raw_control_dict["flags"])
        control.pop("_start_index")
        self._control = control

    @property
    def control(self):
        return self._control

    @property
    def nelements(self):
        return self.control["nelements"]

    @property
    def natoms(self):
        return self.control["natoms"]

    @property
    def ngrid(self):
        return np.asarray(self.control["ngrid"], dtype=int)

    @property
    def lattice(self):
        return np.asarray(self.control["lattvec"], dtype=float) * self.control["lfactor"]

    @property
    def volume(self):
        return np.abs(np.linalg.det(self.lattice))

    @property
    def elements(self):
        return self.control["elements"]

    @property
    def positions(self):
        return np.asarray(self.control["positions"], dtype=float)

    @property
    def scell(self):
        return np.asarray(self.control["scell"], dtype=int)

    @property
    def temperatures(self):
        if "t" in self.control:
            return np.asarray([self.control["t"]], dtype=int)
        else:
            t_min = self.control["t_min"]
            t_max = self.control["t_max"]
            t_step = self.control["t_step"]
            return np.arange(t_min, t_max + 1, t_step, dtype=int)

    @property
    def scalebroad(self):
        return self.control["scalebroad"]

    @property
    def maxiter(self):
        return self.control["maxiter"]

    @property
    def num_sample_process_3ph_phase_space(self):
        return self.control["num_sample_process_3ph_phase_space"]

    @property
    def num_sample_process_3ph(self):
        return self.control["num_sample_process_3ph"]

    @property
    def num_sample_process_4ph_phase_space(self):
        return self.control["num_sample_process_4ph_phase_space"]

    @property
    def num_sample_process_4ph(self):
        return self.control["num_sample_process_4ph"]

    @property
    def convergence(self):
        return self.control["convergence"]

    @property
    def four_phonon(self):
        return self.control["four_phonon"]

    @property
    def four_phonon_iteration(self):
        return self.control["four_phonon_iteration"]

    @property
    def nbands(self):
        return self.natoms * 3

    def _read_frequencies(self):
        omega_path = self.workdir / "BTE.omega"
        self._frequencies = np.loadtxt(omega_path)

    @property
    def frequencies(self):
        return self._frequencies

    def _read_dos(self):
        dos_path = self.workdir / "BTE.dos"
        self._dos = np.loadtxt(dos_path, usecols=1)
        pdos_path = self.workdir / "BTE.pdos"
        self._pdos = np.loadtxt(pdos_path)[:, 1:]

    @property
    def dos(self):
        return self._dos

    @property
    def pdos(self):
        return self._pdos

    def map_flat_quantity_to_q_nu(self, flat_quantity):
        """
        Map nqi*nbands to (nqi, nbands)
        """
        return flat_quantity.reshape((-1, self.nbands), order="F")

    def map_ibz_quantity_to_fbz(self, ibz_quantity):
        """
        IBZ quantity should be (nq_ibz, nu)
        FBZ quantity should be (nq_fbz, nu)
        """
        fbz_quantity = np.zeros((self.nq_fbz, self.nbands), dtype=float)
        for i, idx in enumerate(self.idx_fbz_to_ibz):
            fbz_quantity[idx] = ibz_quantity[i]
        return fbz_quantity

    def _get_idx_fbz_to_ibz(self, qi_fbz_in_ibz):
        """
        For each qpoint in the IBZ, find the corresponding qpoints in the FBZ
        Args:
            qi_fbz_in_ibz: indices for each FBZ qpoint in the IBZ (of length
            nq_fbz)
        Returns:
            idx_fbz_to_ibz: indices for all FBZ qpoints corresponding to each
            IBZ qpoint (of length nq_ibz)
        """
        idx_fbz_to_ibz = []
        for i in range(1, self.nq + 1):
            idx_fbz_to_ibz.append(np.where(qi_fbz_in_ibz == i)[0])
        return idx_fbz_to_ibz

    def _read_qpoints(self):
        qpoints_path = self.workdir / "BTE.qpoints"
        qpoints_file = np.loadtxt(qpoints_path)
        qpoints = qpoints_file[:, 3:]

        qpoints_full_path = self.workdir / "BTE.qpoints_full"
        qpoints_full_file = np.loadtxt(qpoints_full_path)
        qi_fbz = qpoints_full_file[:, 0]
        qi_fbz_in_ibz = qpoints_full_file[:, 1]
        qpoints_fbz = qpoints_full_file[:, 2:]

        self._qpoints = qpoints
        self._qpoints_fbz = qpoints_fbz
        self._idx_fbz_to_ibz = self._get_idx_fbz_to_ibz(qi_fbz_in_ibz)

    @property
    def qpoints(self):
        return self._qpoints

    @property
    def nq(self):
        return len(self.qpoints)

    @property
    def qpoints_fbz(self):
        return self._qpoints_fbz

    @property
    def nq_fbz(self):
        return len(self.qpoints_fbz)

    @property
    def qpoint_degeneracy(self):
        return np.asarray([len(qpoints_fbz) for qpoints_fbz in self.idx_fbz_to_ibz])

    @property
    def qpoint_weights(self):
        return self.qpoint_degeneracy / self.nq_fbz

    @property
    def idx_fbz_to_ibz(self):
        return self._idx_fbz_to_ibz

    def _read_gruneisen(self):
        gruneisen_path = self.workdir / "BTE.gruneisen"
        self._gruneisen = np.loadtxt(gruneisen_path)

    @property
    def gruneisen(self):
        return self._gruneisen

    def _read_group_velocities(self):
        group_velocity_path = self.workdir / "BTE.v"
        group_velocity = np.loadtxt(group_velocity_path)
        group_velocity = group_velocity.reshape(
            (self.nq, self.nbands, 3),
            order="F",
        )
        self._group_velocity = group_velocity
        group_velocity_full_path = self.workdir / "BTE.v_full"
        group_velocity_full = np.loadtxt(group_velocity_full_path)
        group_velocity_full = group_velocity_full.reshape(
            (self.nq_fbz, self.nbands, 3),
            order="F",
        )
        self._group_velocity_full = group_velocity_full

    @property
    def group_velocity(self):
        return self._group_velocity

    @property
    def group_velocity_full(self):
        return self._group_velocity_full

    def _read_phase_space(self):
        p3_path = self.workdir / "BTE.P3"
        self._p3 = np.loadtxt(p3_path)
        p4_path = self.workdir / "BTE.P4"
        if p4_path.exists():
            self._p4 = np.loadtxt(p4_path)

    @property
    def p3(self):
        return self._p3

    @property
    def p4(self):
        return self._p4

    @property
    def temperature_dirs(self):
        """Search the BTE directory for all temperature directories matching T*K"""
        return [
            path
            for path in sorted(
                self.workdir.glob("T*K"), key=lambda d: int(d.name.strip("TK"))
            )
            if path.is_dir()
        ]

    def _read_scattering_rates(self):
        self._rates3 = {}
        self._rates4 = {}
        self._rates = {}
        for temperature_dir in self.temperature_dirs:
            temperature = temperature_dir.name.strip("TK")
            w_3ph_path = temperature_dir / "BTE.w_3ph"
            if w_3ph_path.exists():
                w_3ph = np.loadtxt(w_3ph_path, usecols=1)
                w_3ph = self.map_flat_quantity_to_q_nu(w_3ph)
                if self.scattering_rate_cutoff is not None:
                    w_3ph = np.nan_to_num(w_3ph, nan=0.0)
                    w_3ph = np.where(w_3ph > self.scattering_rate_cutoff, w_3ph, 1e10)
                self._rates3[temperature] = w_3ph
            w_4ph_path = temperature_dir / "BTE.w_4ph"
            if w_4ph_path.exists():
                w_4ph = np.loadtxt(w_4ph_path, usecols=1)
                w_4ph = self.map_flat_quantity_to_q_nu(w_4ph)
                if self.scattering_rate_cutoff is not None:
                    w_4ph = np.nan_to_num(w_4ph, nan=0.0)
                    w_4ph = np.where(w_4ph > self.scattering_rate_cutoff, w_4ph, 1e10)
                self._rates4[temperature] = w_4ph
            w_all_path = temperature_dir / "BTE.w"
            if w_all_path.exists():
                w_all = np.loadtxt(w_all_path, usecols=1)
                w_all = self.map_flat_quantity_to_q_nu(w_all)
                if self.scattering_rate_cutoff is not None:
                    w_all = np.nan_to_num(w_all, nan=0.0)
                    w_all = np.where(w_all > self.scattering_rate_cutoff, w_all, 1e10)
                self._rates[temperature] = w_all

    @property
    def rates(self):
        return self._rates

    @property
    def rates3(self):
        return self._rates3

    @property
    def rates4(self):
        return self._rates4

    @property
    def rates_fbz(self):
        rates_fbz = {}
        for temperature, rates_T in self.rates.items():
            rates_fbz[temperature] = self.map_ibz_quantity_to_fbz(rates_T)
        return rates_fbz

    @property
    def rates3_fbz(self):
        rates3_fbz = {}
        for temperature, rates_T in self.rates3.items():
            rates3_fbz[temperature] = self.map_ibz_quantity_to_fbz(rates_T)
        return rates3_fbz

    @property
    def rates4_fbz(self):
        rates4_fbz = {}
        for temperature, rates_T in self.rates4.items():
            rates4_fbz[temperature] = self.map_ibz_quantity_to_fbz(rates_T)
        return rates4_fbz

    def _get_kappa_as_scalar(self, kappa_tensor):
        if len(kappa_tensor) == 9:
            kappa_tensor = kappa_tensor.reshape(3, 3)
        return np.mean(np.diag(kappa_tensor))

    def _read_kappa(self):
        self._kappa = {}
        self._kappa_tensor = {}
        for temperature_dir in self.temperature_dirs:
            temperature = temperature_dir.name.strip("TK")
            kappa_tensor_path = temperature_dir / "BTE.kappa_tensor"
            kappa_tensor_file = np.loadtxt(kappa_tensor_path)
            kappa_tensor = kappa_tensor_file[-1, 1:]
            kappa = self._get_kappa_as_scalar(kappa_tensor)
            if self.kappa_precision is not None:
                kappa = np.round(kappa, decimals=self.kappa_precision)
                kappa_tensor = np.round(kappa_tensor, decimals=self.kappa_precision)
            self._kappa[temperature] = kappa
            self._kappa_tensor[temperature] = kappa_tensor

    @property
    def kappa(self):
        return self._kappa

    @property
    def kappa_tensor(self):
        return self._kappa_tensor

    def _read_unified_kappa(self):
        unified_results = defaultdict(dict)
        for temperature_dir in self.temperature_dirs:
            temperature = temperature_dir.name.strip("TK")
            unified_kappa_path = temperature_dir / "unifiedkappa.dat"
            if not unified_kappa_path.exists():
                continue
            unified_kappa_tensor_file = np.loadtxt(unified_kappa_path)
            kappa_names = ["unified_kappa_d", "unified_kappa_od", "unified_kappa"]
            for i, kappa_name in enumerate(kappa_names):
                unified_results[kappa_name][temperature] = unified_kappa_tensor_file[i]

        for kappa_name, temperature_dict in unified_results.items():
            for temperature, kappa_tensor in temperature_dict.items():
                kappa_scalar = self._get_kappa_as_scalar(kappa_tensor)
                if self.kappa_precision is not None:
                    kappa_scalar = np.round(kappa_scalar, decimals=self.kappa_precision)
                    kappa_tensor = np.round(kappa_tensor, decimals=self.kappa_precision)
                setattr(self, f"_{kappa_name}", kappa_scalar)
                setattr(self, f"_{kappa_name}_tensor", kappa_tensor)

    @property
    def unified_kappa(self):
        return self._unified_kappa

    @property
    def unified_kappa_d(self):
        return self._unified_kappa_d

    @property
    def unified_kappa_od(self):
        return self._unified_kappa_od

    @property
    def unified_kappa_tensor(self):
        return self._unified_kappa_tensor

    @property
    def unified_kappa_d_tensor(self):
        return self._unified_kappa_d_tensor

    @property
    def unified_kappa_od_tensor(self):
        return self._unified_kappa_od_tensor


if __name__ == "__main__":
    sbte = ShengbteAnalyzer(".")
    print("qpoints analysis")
    print(f"{sbte.nq=} {sbte.qpoints.shape}")
    print(f"{sbte.qpoints=}")
    print(f"{sbte.nq_fbz=} {sbte.qpoints_fbz.shape}")
    print(f"{sbte.qpoints_fbz=}")

    print("Scattering rate analysis")
    for temperature in sbte.temperatures:
        print(f"{temperature}")
        print(f'{sbte.rates[f"{temperature}"].shape}')
        print(f'{sbte.rates[f"{temperature}"]=}')
        print(f'{sbte.rates_fbz[f"{temperature}"].shape}')
        print(f'{sbte.rates_fbz[f"{temperature}"]=}')

    print("Kappa analysis")
    for temperature in sbte.temperatures:
        print(f"{temperature}")
        print(f'{sbte.kappa[f"{temperature}"]}')
        print(f'{sbte.kappa_tensor[f"{temperature}"].shape}')
        print(f'{sbte.kappa_tensor[f"{temperature}"]}')
