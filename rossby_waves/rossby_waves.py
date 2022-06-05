"""Implementation of Rossby waves."""

from cmath import phase
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# scale of sphericity for Rossby waves on Earth
# beta = some-universal-number


def amplitude(wavevector):
    """
    Return amplitude of a Rossby wave.

        Parameters:
            wavevector (np.ndarray) = wavevector np.array([k, l]) of wavenumbers
        
        Returns:
            amplitude (float) = amplitude of Rossby wave
    """
    amplitude = np.exp(-0.1 * wavevector[0]**2 - 0.1 * wavevector[1]**2) * (
        wavevector[0]**2 + wavevector[1]**2)
    return amplitude


def dispersion(wavevector, beta):
    """
    Return frequency from wavevector according to Rossby waves.

        Parameters:
            wavevector (np.ndarray) = wavevector np.array([k, l]) of wavenumbers
            beta (float) = scale of sphericity
        
        Returns:
            omega (float) = frequency of Rossby wave
    """
    omega = -beta * wavevector[0] / (wavevector[0]**2 + wavevector[1]**2)
    return omega


class RossbyWave:
    """
    Class to represent a Rossby wave.

    Attributes
    ----------
    wavevector : tuple
        wavevector (k, l) of wavenumbers
    phase : float
        phase of the wave
    beta : float
        scale of sphericity

    Methods
    -------
    streamfunction(x, t):
        Return streamfunction of Rossby wave.
    """

    def __init__(self, wavevector, phase=0, beta=1):
        self.wavevector = list(wavevector)
        self.k = wavevector[0]
        self.l = wavevector[1]
        self.phase = phase
        self.beta = beta

    def streamfunction(self, x, y, t):
        """
        Return streamfunction of Rossby wave.

            Parameters:
                x (float) = x position coordinate
                y (float) = y position coordinate
                t (float) = time

            Returns:
                psi (float) = streamfunction at x at time t
        """
        psi = amplitude(self.wavevector) * np.cos(
            self.k * x + self.l * y -
            dispersion(self.wavevector, self.beta) * t + self.phase)
        return psi

    def plot_streamfunction(self, xlim=(-1, 1, 100), ylim=(-1, 1, 100), t=0):
        """
        Contour plot of the streamfunction of a Rossby wave.

            Parameters:
                xlim (tuple) = (x start, x end, x points)
                ylim (tuple) = (y start, y end, y points)
                t (float) = time

            Returns:
        """
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        X, Y = np.meshgrid(x, y)
        Z = self.streamfunction(X, Y, t)
        plt.contour(X, Y, Z)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"k={self.k}, l={self.l}, beta={self.beta}")

    def animate_streamfunction(self,
                               xlim=(-1, 1, 100),
                               ylim=(-1, 1, 100),
                               tlim=(0, 1000, 101),
                               filename="streamfunction"):
        """
        Contour plot the streamfunction of a Rossby wave.

            Parameters:
                xlim (tuple) = (x start, x end, x points)
                ylim (tuple) = (y start, y end, y points)
                t (tuple) = (time start, time end, time points)
                filename = file saved as {filename}.gif

            Returns:
        """
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        t = np.linspace(*tlim)
        xx, yy = np.meshgrid(x, y)
        Y, T, X = np.meshgrid(y, t, x)
        fig, ax = plt.subplots(1)
        plot = self.streamfunction(X, Y, T)

        def init_func():
            plt.cla()

        def update_plot(i):
            plt.cla()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f"k={self.k}, l={self.l}, beta={self.beta}")
            plt.contour(xx, yy, plot[i])

        anim = FuncAnimation(fig,
                             update_plot,
                             frames=np.arange(0, len(t)),
                             init_func=init_func)

        writergif = PillowWriter(fps=30)
        anim.save(f'{filename}.gif', writer=writergif)

    def potentialfunction(self, x, y, t, eps=0.1):
        """
        Return streamfunction of Rossby wave.

            Parameters:
                x (float) = x position coordinate
                y (float) = y position coordinate
                t (float) = time
                eps (float) = ratio of stream to potential function

            Returns:
                phi (float) = potentialfunction at x at time t
        """
        phi = eps * self.streamfunction(x, y, t)
        return phi

    def velocity(self,
                 x,
                 y,
                 t,
                 eps=1 / 10,
                 irrotational=False,
                 solenoidal=False):
        """
        Return velocity of Rossby wave at x at time t.

            Parameters:
                x (float) = x position coordinate
                y (float) = y position coordinate
                t (float) = time
                eps (float) = ratio of stream to potential function
                irrotational (bool) = curl-free wave
                solenoidal (bool) = divergence-free wave

            Returns:
                v (np.ndarray) = velocity at x at time t
        """
        # v = (-dpsi/dy, dpsi/dx) + (dphi/dx, dphi/dy)
        # eps*phi = psi = A * exp(k.x - omega * t)
        v = [0, 0]
        if irrotational and solenoidal:
            raise ValueError(
                "Wave cannot be both irrotational and solenoidal.")
        if not solenoidal:
            # no phi
            v[0] += amplitude(self.wavevector) * self.l * np.sin(
                self.k * x + self.l * y -
                dispersion(self.wavevector, self.beta) * t + self.phase)
            v[1] += -amplitude(self.wavevector) * self.k * np.sin(
                self.k * x + self.l * y -
                dispersion(self.wavevector, self.beta) * t + self.phase)
        if not irrotational:
            # no psi
            v[0] += -eps * amplitude(self.wavevector) * self.k * np.sin(
                self.k * x + self.l * y -
                dispersion(self.wavevector, self.beta) * t + self.phase)
            v[1] += -eps * amplitude(self.wavevector) * self.l * np.sin(
                self.k * x + self.l * y -
                dispersion(self.wavevector, self.beta) * t + self.phase)
        return np.array(v)

    def plot_velocity(self,
                      xlim=(-1, 1, 100),
                      ylim=(-1, 1, 100),
                      t=0,
                      density=1):
        """
        Streamplot the velocity of the Rossby wave.
        
            Parameters:
                t (float) = time
                xlim (tuple) = (x start, x end, x points)
                ylim (tuple) = (y start, y end, y points)
                density (float) = density of streamplot arrows
            
            Returns:
        """
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        X, Y = np.meshgrid(x, y)
        u, v = self.velocity(X, Y, t)

        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        ax = fig.add_subplot(gs[0, 0])
        strm = ax.streamplot(X,
                             Y,
                             u,
                             v,
                             color=u + v,
                             density=density,
                             linewidth=1,
                             arrowsize=1.5,
                             arrowstyle='->',
                             cmap='summer')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"k={self.k}, l={self.l}, beta={self.beta}")
        fig.colorbar(strm.lines)

    # figure out beta default
    # find new velocity
    # function plotting tracer path
    # function plotting concentrations


class RossbyOcean():
    """Collection of Rossby waves.

    Attributes
    ----------
    waves : list
        list of RossbyWaves in RossbyOcean
    wavevectors : np.ndarray
        array of wavevectors of RossbyWaves
    phases : np.ndarray
        array of phases
    k : np.ndarray
        array of 1st wavevector components
    l : np.ndarray
        array of 2nd wavevector components
    beta : float
        scale of sphericity

    Methods
    -------
    streamfunction(x, t):
        Return streamfunction of Rossby wave.
    """

    def __init__(self, rossby_waves, beta=1):
        self.waves = rossby_waves
        self.wavevectors = np.array([wave.wavevector for wave in rossby_waves])
        self.phases = np.array([wave.phase for wave in rossby_waves])
        self.k = self.wavevectors[:, 0]
        self.l = self.wavevectors[:, 1]
        self.beta = beta

    def streamfunction(self, x, y, t):
        """
        Return streamfunction of Rossby ocean.

            Parameters:
                x (float) = x position coordinate
                y (float) = y position coordinate
                t (float) = time

            Returns:
                psi (float) = streamfunction at x at time t
        """
        psi = sum(
            amplitude(self.wavevectors.transpose()) *
            np.cos(self.k * x + self.l * y -
                   dispersion(self.wavevectors.transpose(), self.beta) * t +
                   self.phases))
        return psi

    def add_wave(self, wave):
        """
        Add a RossbyWave to the RossbyOcean.
        
            Parameters:
                wave (RossbyWave) = RossbyWave to be added
                
            Returns:
        """
        self.waves.append(wave)
        self.wavevectors = np.array([wave.wavevector for wave in self.waves])
        self.phases = np.array([wave.phase for wave in self.waves])
        self.k = self.wavevectors[:, 0]
        self.l = self.wavevectors[:, 1]
        self = RossbyOcean(self.waves, beta=self.beta)

    def add_random_wave(self, xlim=(-1, 1), ylim=(-1, 1), plim=(0, 2 * np.pi)):
        """Add a RossbyWave to the Rossbyocean with random wavevector.

            Parameters:
                xlim (tuple) = lower and upperbounds of k wavevector component
                ylim (tuple) = lower and upperbounds of l wavevector component
                plim (tuple) = lower and upperbounds of phase

            Returns:
        """
        k = (xlim[1] - xlim[0]) * np.random.random() + xlim[0]
        l = (ylim[1] - ylim[0]) * np.random.random() + ylim[0]
        phase = (plim[1] - plim[0]) * np.random.random() + plim[0]
        self.add_wave(RossbyWave([k, l], phase))

    def random_wavevectors(x, y, xlim=(-5, 5), ylim=(-5, 5)):
        pass

    def normal_wavevectors(xlim=(-5, 5, 10), ylim=(-5, 5, 10)):
        x, y = np.linspace(*xlim), np.linspace(*ylim)
        X, Y = np.meshgrid(x, y)

    # function setting random phases to each wave, use decorators?
    # add random Rossbywave wavevector
    # set up streamfunction
    # set up potential


# function solving Runge-Kutta-4
