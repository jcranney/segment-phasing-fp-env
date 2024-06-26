import pygame
import os
import numpy as np
from .drawable import Drawable


class PSF(Drawable):
    sigma: float = 1.0  # std radians of segment piston per mode
    corr: float = 0.99  # correlation between adjacent samples of state
    nmodes: int = 6  # number of modes in state
    pup_width: int = 64  # width of pupil in pixels
    flux: float = 1e5  # photons per frame
    noise: float = 0.5  # RON in photo-electrons per pixel
    dark: float = 10.0  # photons per pixel of dark current/background

    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.randn(self.nmodes).astype(np.float32)
        self.state *= self.sigma
        self.command = np.zeros(self.nmodes, dtype=np.float32)
        path_pupil = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "images", "pup64.npy")
        self.pupil = np.load(path_pupil)
        path_modes = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "images", "modes64.npy")
        self.modes = np.load(path_modes)
        path_dft = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "images", "dft64.npy")
        self.dft = np.load(path_dft)

        self.ref_max = np.max(np.abs(self.dft.sum(axis=1))**2)

        # update psf
        self.calc_psf()

        # add to le_psf and update counter
        self._le_psf = self.psf * 1.0
        self._iters = 1

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(
            #  pygame.surfarray.make_surface((self.image/2**8).astype(np.uint8)),
            pygame.surfarray.make_surface(self.image),
            (0, 0)
        )

    @staticmethod
    def _rebin(a, factor):
        sh = (a.shape[0]//factor, factor, a.shape[1]//factor, factor)
        return a.reshape(sh).mean(-1).mean(1)

    def calc_psf(self):
        """update PSF based on current state
        """
        phi = np.einsum("ij,i->j", self.modes, self.residual)
        psi = np.exp(1j*phi)
        psi_out = np.einsum("ij,j->i", self.dft, psi)
        psf = np.abs(psi_out)**2
        psf = psf.reshape([int(psf.shape[0]**0.5)]*2)
        self.psf = psf / self.ref_max

    @property
    def image(self) -> np.ndarray:
        img = self._rebin(self.psf, 2)
        img = np.random.poisson(img*self.flux + self.dark)
        img = img + np.random.randn(*img.shape) * self.noise
        img = np.clip(img, 0, 2**16-1)
        return img.astype(np.uint16)

    @property
    def strehl(self) -> float:
        # TODO: this should be changed from "true" Strehl to "estimated" Strehl
        return self.psf.max()

    @property
    def le_psf(self) -> np.ndarray:
        return self._le_psf / self._iters

    @property
    def le_strehl(self) -> float:
        return self.le_psf.max()

    @property
    def residual(self) -> np.ndarray:
        return self.state - self.command

    def act(self, action: np.ndarray) -> None:
        self.command += action
        self.step()

    def step(self) -> None:
        self.state *= self.corr
        self.state += (1-self.corr**2)**0.5 * self.sigma * \
            np.random.randn(self.nmodes)
        self.calc_psf()
        self._le_psf += self.psf
        self._iters += 1
