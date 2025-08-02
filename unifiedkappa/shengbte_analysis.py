from collections import defaultdict
from pathlib import Path

import f90nml
import numpy as np
import phonopy

"""
author: @dgaines2
"""


class ShengBTEAnalyzer:
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
        self._read_group_velocities()
        self._read_phase_space()
        if not self.onlyharmonic:
            self._read_gruneisen()
            self._read_scattering_rates()
            self._read_kappa()
            self._read_unified_kappa()
            self._read_cumulative_kappas()

    def _check_complete(self):
        bte_out_path = self.workdir / "BTE.out"
        if not bte_out_path.exists():
            raise FileNotFoundError(f"No BTE.out file in {self.workdir}... exiting")
        with open(bte_out_path) as fr:
            bte_out = [line.strip() for line in fr.readlines()]
        complete = False
        goodlines = ["normal exit", "onlyharmonic=.true., stopping here"]
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
    def lfactor(self):
        return self.control.get("lfactor", 1.0)

    @property
    def lattice(self):
        return np.asarray(self.control["lattvec"], dtype=float) * self.lfactor

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
    def scell_matrix(self):
        return np.diag(self.scell)

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
        return self.control.get("scalebroad", 1.0)

    @property
    def maxiter(self):
        return self.control.get("maxiter", 1000)

    @property
    def num_sample_process_3ph_phase_space(self):
        return self.control.get("num_sample_process_3ph_phase_space", -1)

    @property
    def num_sample_process_3ph(self):
        return self.control.get("num_sample_process_3ph", -1)

    @property
    def num_sample_process_4ph_phase_space(self):
        return self.control.get("num_sample_process_4ph_phase_space", -1)

    @property
    def num_sample_process_4ph(self):
        return self.control.get("num_sample_process_4ph", -1)

    @property
    def convergence(self):
        return self.control.get("convergence", True)

    @property
    def onlyharmonic(self):
        return self.control.get("onlyharmonic", False)

    @property
    def four_phonon(self):
        return self.control.get("four_phonon", False)

    @property
    def four_phonon_iteration(self):
        return self.control.get("four_phonon_iteration", False)

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

    def _read_group_velocities(self):
        group_velocity_path = self.workdir / "BTE.v"
        group_velocity = np.loadtxt(group_velocity_path)
        group_velocity = group_velocity.reshape(
            (self.nq, self.nbands, 3),
            order="F",
        )
        self._group_velocity = group_velocity
        group_velocity_fbz_path = self.workdir / "BTE.v_full"
        group_velocity_fbz = np.loadtxt(group_velocity_fbz_path)
        group_velocity_fbz = group_velocity_fbz.reshape(
            (self.nq_fbz, self.nbands, 3),
            order="F",
        )
        self._group_velocity_fbz = group_velocity_fbz

    @property
    def group_velocity(self):
        return self._group_velocity

    @property
    def group_velocity_fbz(self):
        return self._group_velocity_fbz

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
    def p3_total(self):
        return np.sum(self.qpoint_degeneracy.reshape(-1, 1) * self.p3)

    @property
    def p4_total(self):
        return np.sum(self.qpoint_degeneracy.reshape(-1, 1) * self.p4)

    def _read_gruneisen(self):
        gruneisen_path = self.workdir / "BTE.gruneisen"
        self._gruneisen = np.loadtxt(gruneisen_path)

    @property
    def gruneisen(self):
        return self._gruneisen

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
        rates_map = {
            "rates3": "BTE.w_3ph",
            "rates4": "BTE.w_4ph",
            "rates_rta": "BTE.w",
            "rates": "BTE.w_final",
        }
        for temperature_dir in self.temperature_dirs:
            temperature = temperature_dir.name.strip("TK")
            for rate_type, rate_filename in rates_map.items():
                rates_file = temperature_dir / rate_filename
                if not rates_file.exists():
                    continue
                rates = np.loadtxt(rates_file, usecols=1)
                rates = self.map_flat_quantity_to_q_nu(rates)
                if self.scattering_rate_cutoff is not None:
                    rates = np.nan_to_num(rates, nan=0.0)
                    rates = np.where(
                        rates > self.scattering_rate_cutoff,
                        rates,
                        1e10,
                    )
                hidden_rate_name = f"_{rate_type}"
                current_rate_attr = getattr(self, hidden_rate_name, None)
                if current_rate_attr is None:
                    current_rate_attr = {}
                    setattr(self, hidden_rate_name, current_rate_attr)
                current_rate_attr[temperature] = rates

    @property
    def rates(self):
        return self._rates

    @property
    def rates_rta(self):
        return self._rates_rta

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
    def rates_rta_fbz(self):
        rates_rta_fbz = {}
        for temperature, rates_T in self.rates_rta.items():
            rates_rta_fbz[temperature] = self.map_ibz_quantity_to_fbz(rates_T)
        return rates_rta_fbz

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

    @staticmethod
    def get_kappa_as_scalar(kappa_tensor):
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
            kappa = self.get_kappa_as_scalar(kappa_tensor)
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
            unified_kappa_path = temperature_dir / f"unifiedkappa.dat"
            if not unified_kappa_path.exists():
                continue
            unified_kappa_tensor_file = np.loadtxt(unified_kappa_path)
            ukappa_names = ["unified_kappa_d", "unified_kappa_od", "unified_kappa"]
            for i, ukappa_name in enumerate(ukappa_names):
                unified_results[ukappa_name][temperature] = unified_kappa_tensor_file[i]

        for ukappa_name, temperature_dict in unified_results.items():
            for temperature, ukappa_tensor in temperature_dict.items():
                ukappa_scalar = self.get_kappa_as_scalar(ukappa_tensor)
                if self.kappa_precision is not None:
                    ukappa_scalar = np.round(ukappa_scalar, decimals=self.kappa_precision)
                    ukappa_tensor = np.round(ukappa_tensor, decimals=self.kappa_precision)

                ukappa_attr_name = f"_{ukappa_name}"
                current_ukappa_attr = getattr(self, ukappa_attr_name, None)
                if current_ukappa_attr is None:
                    current_ukappa_attr = {}
                    setattr(self, ukappa_attr_name, current_ukappa_attr)
                current_ukappa_attr[temperature] = ukappa_scalar
                ukappa_tensor_attr_name = f"_{ukappa_name}_tensor"
                current_ukappa_tensor_attr = getattr(self, ukappa_tensor_attr_name, None)
                if current_ukappa_tensor_attr is None:
                    current_ukappa_tensor_attr = {}
                    setattr(self, ukappa_tensor_attr_name, current_ukappa_tensor_attr)
                current_ukappa_tensor_attr[temperature] = ukappa_tensor

    @property
    def unified_kappa(self):
        return self._unified_kappa

    @property
    def unified_kappa_D(self):
        return self._unified_kappa_d

    @property
    def unified_kappa_OD(self):
        return self._unified_kappa_od

    @property
    def unified_kappa_tensor(self):
        return self._unified_kappa_tensor

    @property
    def unified_kappa_D_tensor(self):
        return self._unified_kappa_d_tensor

    @property
    def unified_kappa_OD_tensor(self):
        return self._unified_kappa_od_tensor

    def _read_cumulative_kappas(self):
        ckappa_map = {
            "ckappa_tensor_vs_mfp": {
                "fname": "BTE.cumulative_kappa_tensor",
                "xname": "mfp",
            },
            "ckappa_tensor_vs_omega": {
                "fname": "BTE.cumulative_kappaVsOmega_tensor",
                "xname": "omega",
            },
        }
        for temperature_dir in self.temperature_dirs:
            temperature = temperature_dir.name.strip("TK")
            for ckappa_name, ckappa_info in ckappa_map.items():
                ckappa_file = temperature_dir / ckappa_info["fname"]
                ckappa_data = np.loadtxt(ckappa_file)

                hidden_ckappa_name = f"_{ckappa_name}"
                current_ckappa_attr = getattr(self, hidden_ckappa_name, None)
                if current_ckappa_attr is None:
                    current_ckappa_attr = {}
                    setattr(self, hidden_ckappa_name, current_ckappa_attr)
                current_ckappa_attr[temperature] = {
                    ckappa_info["xname"]: ckappa_data[:, 0],
                    "ckappa": ckappa_data[:, 1:],
                }

    @property
    def ckappa_vs_omega(self):
        ckappa_vs_omega = {}
        for temperature, ckappa_dict in self.ckappa_tensor_vs_omega.items():
            ckappa_vs_omega[temperature] = {
                "omega": ckappa_dict["omega"],
                "ckappa": np.apply_along_axis(
                    self.get_kappa_as_scalar,
                    1,
                    ckappa_dict["ckappa"],
                ),
            }
        return ckappa_vs_omega

    @property
    def ckappa_tensor_vs_omega(self):
        return self._ckappa_tensor_vs_omega

    @property
    def ckappa_vs_mfp(self):
        ckappa_vs_mfp = {}
        for temperature, ckappa_dict in self.ckappa_tensor_vs_mfp.items():
            ckappa_vs_mfp[temperature] = {
                "mfp": ckappa_dict["mfp"],
                "ckappa": np.apply_along_axis(
                    self.get_kappa_as_scalar,
                    1,
                    ckappa_dict["ckappa"],
                ),
            }
        return ckappa_vs_mfp

    @property
    def ckappa_tensor_vs_mfp(self):
        return self._ckappa_tensor_vs_mfp


def unwrap_qpoints(qpoints):
    """
    Convert fractional qpoint coordinates into ShengBTE format (such that there
    are no negative values)
    """
    n_qpoints = len(qpoints)
    qpoints = qpoints.flatten()
    qpoints = np.where(qpoints >= 0, qpoints, 1 - np.abs(qpoints))
    qpoints = qpoints.reshape(n_qpoints, -1)
    return qpoints


if __name__ == "__main__":
    sbte = ShengBTEAnalyzer(".")
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
