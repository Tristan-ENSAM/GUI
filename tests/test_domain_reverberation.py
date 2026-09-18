# -*- coding: utf-8 -*-
"""
Reverberation ceiling on the Eulerian domain (gui.core.domain_sizing).

Replaces the old `diagonal_limit(elem_size) = 90.6 * elem_size` ceiling, which
came from requiring a non-empty mass-scaling window (hence from the Abaqus
RECOMMENDATION fc*dt > 1e-3), not from reverberation. The physical constraint
is the cavity mode f = c_eff/(2 L), measured on three campaigns; see
`reverberation_frequency`'s docstring for the table and its provenance.
"""
from __future__ import annotations

import math
import pytest

from gui.core.domain_sizing import (
    DomainDims, REVERB_MARGIN_K, IIR_ADVISED_MIN_RATIO, NYQUIST_MAX_RATIO,
    diagonal, dilatational_wave_speed, effective_wave_speed,
    reverberation_frequency, reverberation_diagonal_limit,
    filter_ratio_check, config_diagonal_limit, config_filter_check,
    config_mass_scaling,
)

# Production Ti-6Al-4V point, Abaqus mm-t-s units (E in MPa, rho in t/mm^3).
E, NU, RHO = 113800.0, 0.342, 4.43e-9
MS = 1000.0
FC = 25600.0            # field cutoff, the DIC chain (model_config.py:162)


class TestWaveSpeeds:
    def test_dilatational_speed_matches_the_reported_value(self):
        # 6313 m/s = 6.313e6 mm/s, reported with the campaign.
        assert dilatational_wave_speed(E, NU, RHO) == pytest.approx(6.3134e6,
                                                                   rel=1e-3)

    def test_same_as_model_config_lame_form(self):
        # sqrt(E(1-nu)/(rho(1+nu)(1-2nu))) == sqrt((lambda+2mu)/rho), the form
        # used in ModelConfig.initial_stable_dt (model_config.py:811-813).
        lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
        mu = E / (2.0 * (1.0 + NU))
        assert dilatational_wave_speed(E, NU, RHO) == pytest.approx(
            math.sqrt((lam + 2.0 * mu) / RHO), rel=1e-12)

    def test_effective_speed_is_c_over_sqrt_ms(self):
        # 6313 m/s at ms = 1000 -> 199.65 m/s.
        c_eff = effective_wave_speed(dilatational_wave_speed(E, NU, RHO), MS)
        assert c_eff == pytest.approx(199.65e3, rel=2e-3)   # mm/s

    @pytest.mark.parametrize("bad", [
        dict(E=0.0, nu=NU, rho=RHO),
        dict(E=E, nu=NU, rho=0.0),
        dict(E=E, nu=0.5, rho=RHO),      # nu = 0.5 -> incompressible, undefined
        dict(E=E, nu=-2.0, rho=RHO),
    ])
    def test_unusable_input_returns_zero(self, bad):
        assert dilatational_wave_speed(**bad) == 0.0


class TestCavityModel:
    """f = c_eff/(2 L) against the three measured campaigns (ms = 1000,
    identical mesh and source, filters OFF, peak of the detrended ALLKE
    spectrum). Requested accuracy: better than 8 %."""

    MEASURED = [
        (0.42, 241.6e3),
        (0.84, 123.1e3),
        (1.26, 85.3e3),
    ]

    @pytest.mark.parametrize("L_mm,f_measured", MEASURED)
    def test_predicts_each_campaign_within_8pct(self, L_mm, f_measured):
        c_eff = effective_wave_speed(dilatational_wave_speed(E, NU, RHO), MS)
        f_pred = reverberation_frequency(c_eff, L_mm)
        assert abs(f_pred - f_measured) / f_measured < 0.08

    def test_the_law_is_one_over_L(self):
        # Doubling the diagonal halves the peak, exactly.
        c_eff = effective_wave_speed(dilatational_wave_speed(E, NU, RHO), MS)
        assert (reverberation_frequency(c_eff, 0.42)
                / reverberation_frequency(c_eff, 0.84)) == pytest.approx(2.0)

    def test_fitted_exponent_is_close_to_minus_one(self):
        # log-log fit of the MEASURED points: the reported exponent is -0.951.
        xs = [math.log(L) for L, _ in self.MEASURED]
        ys = [math.log(f) for _, f in self.MEASURED]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                 / sum((x - mx) ** 2 for x in xs))
        assert slope == pytest.approx(-0.951, abs=0.01)


class TestDiagonalLimit:
    def test_reference_point_is_1_300_mm(self):
        L_max = reverberation_diagonal_limit(
            filter_cutoff_hz=FC, mass_scaling=MS, E=E, nu=NU, rho=RHO,
            margin_k=3.0)
        assert L_max == pytest.approx(1.300, rel=2e-3)

    def test_c_d_and_enu_rho_forms_agree(self):
        c_d = dilatational_wave_speed(E, NU, RHO)
        a = reverberation_diagonal_limit(FC, MS, c_d=c_d)
        b = reverberation_diagonal_limit(FC, MS, E=E, nu=NU, rho=RHO)
        assert a == pytest.approx(b, rel=1e-12)

    def test_at_the_limit_the_mode_sits_exactly_k_times_the_cutoff(self):
        # This is the definition: L_max is where f_reverb == k * fc.
        k = 3.0
        L_max = reverberation_diagonal_limit(FC, MS, E=E, nu=NU, rho=RHO,
                                             margin_k=k)
        c_eff = effective_wave_speed(dilatational_wave_speed(E, NU, RHO), MS)
        assert reverberation_frequency(c_eff, L_max) == pytest.approx(k * FC)

    def test_k_is_a_parameter_and_defaults_to_three(self):
        assert REVERB_MARGIN_K == 3.0
        base = reverberation_diagonal_limit(FC, MS, E=E, nu=NU, rho=RHO)
        assert base == pytest.approx(
            reverberation_diagonal_limit(FC, MS, E=E, nu=NU, rho=RHO,
                                         margin_k=3.0))
        # The ceiling is inversely proportional to k: halving k doubles it.
        loose = reverberation_diagonal_limit(FC, MS, E=E, nu=NU, rho=RHO,
                                             margin_k=1.5)
        assert loose == pytest.approx(2.0 * base)

    def test_ceiling_shrinks_as_sqrt_ms(self):
        a = reverberation_diagonal_limit(FC, 1000.0, E=E, nu=NU, rho=RHO)
        b = reverberation_diagonal_limit(FC, 4000.0, E=E, nu=NU, rho=RHO)
        assert a / b == pytest.approx(2.0)

    def test_is_the_mass_scaling_bound_solved_for_L(self):
        # ModelConfig.mass_scaling_bounds states ms < (c_d/(2 L k fc))^2
        # (model_config.py:834). At L = L_max that bound must equal ms itself.
        c_d = dilatational_wave_speed(E, NU, RHO)
        L_max = reverberation_diagonal_limit(FC, MS, c_d=c_d, margin_k=3.0)
        assert (c_d / (2.0 * L_max * 3.0 * FC)) ** 2 == pytest.approx(MS)

    def test_old_90p6_ceiling_was_about_7x_too_tight(self):
        # FACT being corrected: at the production mesh h = 0.002 mm the old
        # ceiling was 90.6*h = 0.1812 mm, against 1.300 mm physically.
        old = 90.6 * 0.002
        new = reverberation_diagonal_limit(FC, MS, E=E, nu=NU, rho=RHO)
        assert new / old == pytest.approx(7.17, rel=0.02)

    @pytest.mark.parametrize("kw", [
        dict(filter_cutoff_hz=0.0, mass_scaling=MS, E=E, nu=NU, rho=RHO),
        dict(filter_cutoff_hz=FC, mass_scaling=0.0, E=E, nu=NU, rho=RHO),
        dict(filter_cutoff_hz=FC, mass_scaling=MS, E=0.0, nu=NU, rho=RHO),
        dict(filter_cutoff_hz=FC, mass_scaling=MS, margin_k=0.0, E=E, nu=NU,
             rho=RHO),
        dict(filter_cutoff_hz=FC, mass_scaling=MS),          # nothing given
    ])
    def test_unusable_input_returns_zero(self, kw):
        assert reverberation_diagonal_limit(**kw) == 0.0


class TestFilterRatioCheck:
    """One HARD control (Nyquist), one ADVISORY (the 1e-3 recommendation)."""

    def test_production_ratio_is_below_1e3_and_only_warns(self):
        # Reported production ratio: 5.447e-4, i.e. below the recommendation.
        dt = 5.447e-4 / FC
        chk = filter_ratio_check(FC, dt)
        assert chk.ratio == pytest.approx(5.447e-4, rel=1e-9)
        assert chk.iir_advised_ok is False        # advisory violated
        assert chk.nyquist_ok is True             # but nothing is blocked
        assert "not blocking" in chk.message
        # The documented remedy must be spelled out.
        assert "two-stage" in chk.message and "post-processing" in chk.message

    def test_nyquist_is_the_hard_control(self):
        dt = 0.5 / FC                             # fc == half the sampling rate
        chk = filter_ratio_check(FC, dt)
        assert chk.nyquist_ok is False
        assert "BLOCKING" in chk.message
        assert chk.ratio == pytest.approx(NYQUIST_MAX_RATIO)

    def test_comfortable_ratio_passes_both(self):
        dt = 1e-2 / FC
        chk = filter_ratio_check(FC, dt)
        assert chk.nyquist_ok and chk.iir_advised_ok

    def test_boundaries_are_strict(self):
        exactly_advised = filter_ratio_check(FC, IIR_ADVISED_MIN_RATIO / FC)
        assert exactly_advised.iir_advised_ok is False   # strict ">"
        assert exactly_advised.nyquist_ok is True

    def test_not_computable_on_bad_input(self):
        chk = filter_ratio_check(FC, 0.0)
        assert chk.computable is False
        assert chk.nyquist_ok is False and chk.iir_advised_ok is False


class TestConfigAdapters:
    """The adapters read a real ModelConfig by attribute (duck-typed)."""

    def _cfg(self, ms=MS, fc=FC, elem=0.002, enabled=True):
        from gui.core.model_config import ModelConfig
        c = ModelConfig()
        c.euler_material.update({"E": E, "nu": NU, "rho": RHO})
        c.elem_size = elem
        c.step.mass_scaling_enabled = enabled
        c.step.mass_scaling_factor = ms
        c.step.output_filter_cutoff_hz = fc
        return c

    def test_mass_scaling_is_one_when_disabled(self):
        assert config_mass_scaling(self._cfg(enabled=False)) == 1.0
        assert config_mass_scaling(self._cfg()) == pytest.approx(MS)

    def test_config_ceiling_matches_the_reference_point(self):
        assert config_diagonal_limit(self._cfg()) == pytest.approx(1.300,
                                                                   rel=2e-3)

    def test_config_ceiling_honours_k(self):
        base = config_diagonal_limit(self._cfg())
        assert config_diagonal_limit(self._cfg(), margin_k=1.5) == \
            pytest.approx(2.0 * base)

    def test_config_ceiling_none_when_material_is_unusable(self):
        c = self._cfg()
        c.euler_material["rho"] = 0.0
        assert config_diagonal_limit(c) is None

    def test_config_ceiling_none_on_a_stub_without_step(self):
        class _Stub:
            euler_material = {"E": E, "nu": NU, "rho": RHO}
        assert config_diagonal_limit(_Stub()) is None

    def test_config_filter_check_reproduces_the_production_ratio(self):
        # dt = dt0 * sqrt(ms); dt0 at h = 2 um is 1.7230e-10 s (measured .sta,
        # tests/test_mass_scaling_bounds.py:34).
        c = self._cfg()
        chk = config_filter_check(c)
        assert chk is not None and chk.computable
        expected = FC * c.initial_stable_dt() * math.sqrt(MS)
        assert chk.ratio == pytest.approx(expected, rel=1e-12)
        # ~1.394e-4: below the recommendation, far below Nyquist.
        assert chk.iir_advised_ok is False and chk.nyquist_ok is True

    def test_config_filter_check_none_on_a_stub(self):
        class _Stub:
            pass
        assert config_filter_check(_Stub()) is None


class TestDiagonalUnchanged:
    def test_diagonal_is_still_the_plain_hypotenuse(self):
        d = DomainDims(h_wp=0.03, h_void=0.04, l_wp=0.06, l_void=0.02)
        assert diagonal(d) == pytest.approx(math.hypot(0.08, 0.07))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
