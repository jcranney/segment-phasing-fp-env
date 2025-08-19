# segment_phasing_fp_env/psf_autodiff.py
from __future__ import annotations

import os
import numpy as np
import jax.numpy as jnp
from jax import random, jit, jacfwd, jacrev
from .drawable import Drawable


#Factory function: bind `side` and return JIT pure function
def _make_calc_psf_core_jax(side: int):
    @jit
    def f(modes, dft, ref_max, residual):
        phi = jnp.einsum("ij,i->j", modes, residual, optimize=True)
        psi = jnp.exp(1j * phi)
        psi_out = jnp.einsum("ij,j->i", dft, psi, optimize=True)
        psf = jnp.abs(psi_out) ** 2
        return psf.reshape((side, side)) / ref_max
    return f


class PSFAutoDiff(Drawable):
    """
    PSF model with JAX backend (autodiff friendly).
    Provides Jacobian/Hessian of measurements.
    """
    sigma: float = 1.0
    corr: float = 0.99
    nmodes: int = 6
    flux: float = 5e4
    noise: float = 0.5
    dark: float = 10.0
    ideal: bool

    def __init__(self, seed: int | None = None, ideal: bool = False):
        self.ideal = ideal
        key = random.PRNGKey(0 if seed is None else seed)

        self.state = random.normal(key, (self.nmodes,)) * self.sigma
        self.command = jnp.zeros(self.nmodes)
        base = os.path.dirname(os.path.realpath(__file__))
        self.pupil = jnp.array(np.load(os.path.join(base, "images", "pup64.npy")))
        self.modes = jnp.array(np.load(os.path.join(base, "images", "modes64.npy")))

        dft_np = np.load(os.path.join(base, "images", "dft64.npy"))
        rows = int(dft_np.shape[0])
        side = int(np.sqrt(rows))
        if side * side != rows:
            raise ValueError(f"DFT rows {rows} is not a perfect square.")
        self.side: int = side
        self.dft = jnp.array(dft_np).astype(jnp.complex64)
        self.ref_max = jnp.max(jnp.abs(self.dft.sum(axis=1)) ** 2)
        self._calc_psf = _make_calc_psf_core_jax(self.side)
        self.calc_psf()
        self._le_psf = self.psf * 1.0
        self._iters = 1

    @property
    def residual(self) -> jnp.ndarray:
        return self.state + self.command

    @property
    def strehl(self) -> float:
        return float(jnp.max(self.psf))

    @staticmethod
    def _rebin(a: jnp.ndarray, factor: int) -> jnp.ndarray:
        h, w = a.shape
        sh = (h // factor, factor, w // factor, factor)
        return a.reshape(sh).mean(-1).mean(1)

    def calc_psf(self, *, residual: jnp.ndarray | None = None):
        r = self.residual if residual is None else residual
        self.psf = self._calc_psf(self.modes, self.dft, self.ref_max, r)

    @property
    def image(self):
        img = self._rebin(self.psf, 2) * self.flux
        if self.ideal:
            return np.asarray(img, dtype=np.float32)

        img_np = np.asarray(img, dtype=np.float32)
        img_np = np.random.poisson(img_np + self.dark).astype(np.float32)
        img_np = img_np + np.random.randn(*img_np.shape).astype(np.float32) * self.noise
        img_np = np.clip(img_np, 0, 2**16 - 1).astype(np.uint16)
        return img_np

    def draw(self, window):
        import pygame
        window.blit(pygame.surfarray.make_surface(self.image), (0, 0))

    def act(self, action: jnp.ndarray) -> None:
        self.command = self.command + action
        self.step()

    def step(self) -> None:
        self.state = self.state * self.corr
        self.calc_psf()
        self._le_psf = self._le_psf + self.psf
        self._iters += 1

    def poke(self, residual: jnp.ndarray):
        self.calc_psf(residual=residual)
        return self.image

    #Autodiff interface
    def poke_flat(self, residual: jnp.ndarray) -> jnp.ndarray:
        psf = self._calc_psf(self.modes, self.dft, self.ref_max, residual)
        img = self._rebin(psf, 2) * self.flux
        return img.reshape(-1)

    def measurement_jacobian(self, x: jnp.ndarray | None = None) -> jnp.ndarray:
        x = self.residual if x is None else x
        return jacfwd(self.poke_flat)(x)

    def hessian_wrt_state(self, x: jnp.ndarray | None = None) -> jnp.ndarray:
        x = self.residual if x is None else x
        return jacfwd(jacrev(self.poke_flat))(x)
