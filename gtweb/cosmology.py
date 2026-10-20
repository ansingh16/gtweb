"""
Linear-theory cosmology used by the halo model (Sheth-Tormen mass function).

The Eisenstein & Hu (1998) no-wiggle transfer function and a sigma8-normalised LCDM
linear P(k), for the Planck / MDPL2 cosmology used in the paper.  Pure numpy/scipy.
"""
import numpy as np
from scipy.integrate import quad

# MDPL2 / Planck cosmology (z = 0, growth D = 1)
Om, Ob, hubble, ns, sigma8, Tcmb = 0.307, 0.048, 0.6777, 0.96, 0.8228, 2.7255
Ol = 1.0 - Om                # flat LCDM dark-energy density (Omega_Lambda)
H0 = 100.0                   # km/s / (Mpc/h)  -- h-units, so H0 = 100 exactly

RHO_CRIT = 2.775e11          # Msun/h / (Mpc/h)^3
RHO_M = Om * RHO_CRIT        # mean matter density
DELTA_C = 1.686              # linear collapse barrier
DELTA_VIR = 200.0            # virial overdensity wrt mean


def growth_rate_f1(Om_=Om):
    """First-order linear growth rate f1 = dlnD1/dlna ~ Om^0.55 (z=0)."""
    return Om_ ** 0.55


def growth_rate_f2(Om_=Om):
    """Second-order (2LPT) growth rate f2 = dlnD2/dlna ~ 2 Om^(6/11) (Bouchet et al. 1995)."""
    return 2.0 * Om_ ** (6.0 / 11.0)


# ------------------------------------------------------------------ background + linear growth (flat LCDM)
def E(a):
    """Dimensionless Hubble rate H(a)/H0 = sqrt(Om a^-3 + Ol) for flat LCDM."""
    a = np.asarray(a, float)
    return np.sqrt(Om * a ** -3 + Ol)


def H_of_a(a):
    """H(a) in km/s / (Mpc/h)  (H0 = 100 in h-units)."""
    return H0 * E(a)


def Omega_m_a(a):
    """Matter density parameter at scale factor a."""
    a = np.asarray(a, float)
    return Om * a ** -3 / (Om * a ** -3 + Ol)


def growth_rate_f(a):
    """Linear growth rate f = dlnD/dlna ~ Omega_m(a)^0.55 (Linder 2005) -- reduces to f1 at a=1."""
    return Omega_m_a(a) ** 0.55


def _growth_unnorm(a):
    """Unnormalised linear growth D(a) ~ H(a) int_0^a da' / (a' H(a'))^3  (Heath 1977)."""
    a = float(a)
    integ = lambda ap: 1.0 / (ap * E(ap)) ** 3
    return E(a) * quad(integ, 1e-8, a, limit=200)[0]


def growth_D(z=0.0):
    """Linear growth factor D(z) normalised to D(z=0)=1 (flat LCDM, growing mode)."""
    a = 1.0 / (1.0 + np.asarray(z, float))
    d0 = _growth_unnorm(1.0)
    if a.ndim == 0:
        return _growth_unnorm(float(a)) / d0
    return np.array([_growth_unnorm(float(ai)) for ai in a.ravel()]).reshape(a.shape) / d0


def T_nowiggle(k):
    """Eisenstein & Hu 1998 no-wiggle transfer T(k), k in h/Mpc (nbodykit NoWiggleEH convention)."""
    omh2, obh2, fb, theta = Om * hubble ** 2, Ob * hubble ** 2, Ob / Om, Tcmb / 2.7
    s = hubble * 44.5 * np.log(9.83 / omh2) / np.sqrt(1.0 + 10.0 * obh2 ** 0.75)      # sound horizon [Mpc/h]
    ag = 1 - 0.328 * np.log(431 * omh2) * fb + 0.38 * np.log(22.3 * omh2) * fb ** 2
    geff = Om * hubble * (ag + (1 - ag) / (1 + (0.43 * k * s) ** 4))
    q = k * theta ** 2 / geff
    L0 = np.log(2 * np.e + 1.8 * q); C0 = 14.2 + 731.0 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q * q)


def Plin_unnorm(k):
    return k ** ns * T_nowiggle(k) ** 2


def _sigma2(R, norm=1.0):
    def integ(lnk):
        k = np.exp(lnk); x = k * R
        w = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
        return (k ** 3 * norm * Plin_unnorm(k)) / (2 * np.pi ** 2) * w ** 2      # dsigma2/dlnk
    return quad(integ, np.log(1e-4), np.log(50.0), limit=200)[0]


def linear_Pk(k):
    """LCDM linear P(k) [ (Mpc/h)^3 ], normalised so sigma(8 Mpc/h) = sigma8."""
    A = sigma8 ** 2 / _sigma2(8.0, 1.0)
    return A * Plin_unnorm(k)
