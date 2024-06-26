from typing import Any, Dict, List, SupportsFloat, Tuple
from gymnasium.core import ActType, ObsType, RenderFrame

import functools

import gymnasium as gym
import numpy as np
import pygame

from gymnasium.spaces import Box

from .psf import PSF


class SegmentPhasingFPEnv(gym.Env):
    action_space = Box(low=-2.0, high=2.0, shape=(6,))
    """
    The Space object corresponding to valid actions, all valid actions should
    be contained with the space.
    """

    observation_space = Box(low=0, high=2**16-1, shape=(20, 20),
                            dtype=np.uint16)
    """
    The Space object corresponding to valid observations, all valid
    observations should be contained with the space. It is static across all
    instances.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    """
    The metadata of the environment containing rendering modes, rendering fps,
    etc.
    """

    def __init__(self, render_mode: str | None = None):
        self.render_mode = render_mode

        self._psf = None

        self._surface = None
        self._clock = None
        if self.render_mode == "human":
            self._clock = pygame.time.Clock()

        self._last_action = None
        self._upscale = 32  # upscale image output for display purposes
        self._score: float = 0.0  # score for the episode
        self.max_steps: int = 20  # maximum steps per episode
        self.step_counter: int = 0  # steps in this episode so far

    @property
    def observation(self) -> ObsType:
        """The observation is the most recent frame computed from the
        environment. This alone is not enough to estimate the segment phase,
        but a successful solution could utilise a time-series of this data
        along with the command sequence applied.
        """
        return self._image

    @property
    def reward(self) -> SupportsFloat:
        if not self.terminated:
            return self._psf.strehl
        return 0.0

    @property
    def terminated(self) -> bool:
        """The termination logic is that any of the following conditions
        imply failure:
         - Strehl less than 10%,
         - integrated command greater than 10.0 radians for any more,
         - too much time elapsed.
         """
        return any([
            self._psf.strehl < 0.1,
            np.max(np.abs(self._psf.command)) > 10.0,
            self.step_counter >= self.max_steps,
        ])

    @property
    def truncated(self) -> bool:
        return False

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "le_psf": self._psf.le_psf,
            "le_strehl": self._psf.le_strehl,
            "residual_modes": self._psf.residual,
            "last_action": self._last_action,
            "score": self._score
        }

    def step(self, action: ActType) -> \
            tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Run one timestep of the environment’s dynamics using the agent actions.

        When the end of an episode is reached (`terminated` or `truncated`), it
        is necessary to call `reset()` to reset this environment’s state for
        the next episode.

        Parameters:
        - action (ActType): an action provided by the agent to update the
        environment state.

        Returns:
        - observation (ObsType): An element of the environment’s
        `observation_space` as the next observation due to the agent actions.
        An example is a numpy array containing the positions and velocities of
        the pole in CartPole.

        - reward (SupportsFloat): The reward as a result of taking the action.

        - terminated (bool): Whether the agent reaches the terminal state (as
        defined under the MDP of the task) which can be positive or negative.
        An example is reaching the goal state or moving into the lava from the
        Sutton and Barton, Gridworld. If true, the user needs to call
        `reset()`.

        - truncated (bool): Whether the truncation condition outside the scope
        of the MDP is satisfied. Typically, this is a timelimit, but could also
        be used to indicate an agent physically going out of bounds. Can be
        used to end the episode prematurely before a terminal state is reached.
        If true, the user needs to call `reset()`.

        - info (dict): Contains auxiliary diagnostic information (helpful for
        debugging, learning, and logging). This might, for instance, contain:
        metrics that describe the agent’s performance state, variables that are
        hidden from observations, or individual reward terms that are combined
        to produce the total reward. In OpenAI Gym <v26, it contains
        `TimeLimit.truncated` to distinguish truncation and termination,
        however this is deprecated in favour of returning terminated and
        truncated variables.
        """
        self._prev_image = self._image
        self._psf.act(action)
        self._image = self._psf.image

        self._score += self._psf.strehl

        if self.render_mode == "human":
            self.render()

        self.step_counter += 1

        return self.observation, self.reward, self.terminated, \
            self.truncated, self.info

    def reset(self, *, seed: int | None = None,
              options: Dict[str, Any] | None = None) \
            -> Tuple[ObsType, Dict[str, Any]]:
        """
        Resets the environment to an initial internal state, returning an
        initial observation and info.

        This method generates a new starting state often with some randomness
        to ensure that the agent explores the state space and learns a
        generalised policy about the environment. This randomness can be
        controlled with the seed parameter otherwise if the environment already
        has a random number generator and `reset()` is called with `seed=None`,
        the RNG is not reset.

        Therefore, `reset()` should (in the typical use case) be called with a
        seed right after initialization and then never again.

        For Custom environments, the first line of `reset()` should be
        `super().reset(seed=seed)` which implements the seeding correctly.

        Parameters:
        - seed (optional int): The seed that is used to initialize the
        environment’s PRNG (`np_random`). If the environment does not already
        have a PRNG and `seed=None` (the default option) is passed, a seed will
        be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
        However, if the environment already has a PRNG and `seed=None` is
        passed, the PRNG will not be reset. If you pass an integer, the PRNG
        will be reset even if it already exists. Usually, you want to pass an
        integer right after the environment has been initialized and then never
        again.

        - options (optional dict): Additional information to specify how the
        environment is reset (optional, depending on the specific environment).

        Returns:
        - observation (ObsType): Observation of the initial state. This will be
        an element of `observation_space` (typically a numpy array) and is
        analogous to the observation returned by `step()`.

        - info (dictionary): This dictionary contains auxiliary information
        complementing observation. It should be analogous to the info returned
        by `step()`.
        """

        super().reset(seed=seed)

        self._psf = PSF()
        self._prev_image = self._psf.image
        self._psf.step()
        self._image = self._psf.image
        self._surface = None

        self._last_action = None
        self._score = 0

        self.step_counter = 0

        if self.render_mode is not None:
            self.render()

        return self.observation, self.info

    def render(self) -> RenderFrame | List[RenderFrame] | None:
        """
        Compute the render frames as specified by render_mode during the
        initialization of the environment.

        The environment’s metadata render modes
        (`env.metadata[“render_modes”]`) should contain the possible ways to
        implement the render modes. In addition, list versions for most render
        modes is achieved through `gymnasium.make` which automatically applies
        a wrapper to collect rendered frames.

        Note: As the render_mode is known during `__init__`, the objects used
        to render the environment state should be initialised in `__init__`.

        By convention, if the render_mode is:
        - None (default): no render is computed.

        - "human": The environment is continuously rendered in the current
        display or terminal, usually for human consumption. This rendering
        should occur during `step()` and `render()` doesn’t need to be called.
        Returns None.

        - "rgb_array": Return a single frame representing the current state of
        the environment. A frame is a `np.ndarray` with shape (x, y, 3)
        representing RGB values for an x-by-y pixel image.
        """
        array = np.tile(
            (self.observation**0.5)[:, :, None].astype(np.uint8),
            [1, 1, 3]
        )
        array = array.repeat(self._upscale, axis=0).repeat(
            self._upscale, axis=1)

        if self._surface is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Segment Phasing FP")
                self._surface = pygame.display.set_mode(self._shape)
            elif self.render_mode == "rgb_array":
                self._surface = pygame.Surface(self._shape)
                return array

        assert self._surface is not None, \
            "Something went wrong with pygame. This should never happen."

        self._surface.blit(
            pygame.surfarray.make_surface(array),
            (0, 0)
        )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return array

    @property
    @functools.cache
    def _width(self) -> int:
        return self.observation_space.shape[1]*self._upscale

    @property
    @functools.cache
    def _height(self) -> int:
        return self.observation_space.shape[0]*self._upscale

    @property
    @functools.cache
    def _shape(self) -> Tuple[int, int]:
        return self._width, self._height

    def close(self) -> None:
        """
        After the user has finished using the environment, close contains the
        code necessary to "clean up" the environment.

        This is critical for closing rendering windows, database or HTTP
        connections.
        """

        if self._surface is not None:
            pygame.display.quit()
            pygame.quit()
