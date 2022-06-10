"""Implementation of Rossby waves."""
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from math import floor

# scale of sphericity for Rossby waves in Ocean
beta = 2e-11


def amplitude(wavevector):
    """
    Return amplitude of a Rossby wave.

    Parameters
    ----------
    wavevector : np.ndarray
        wavevector np.array([k, l]) of wavenumbers
        
    Returns
    -------
    amplitude : float
        amplitude of Rossby wave
    """
    amplitude = 1 / 6 * np.exp(-wavevector[0]**2 / 25 - wavevector[1]**2 /
                               25) * (wavevector[0]**2 + wavevector[1]**2)
    return amplitude


def dispersion(wavevector, beta=beta):
    """
    Return frequency from wavevector according to Rossby waves.

    Parameters
    ----------
    wavevector : np.ndarray
        wavevector np.array([k, l]) of wavenumbers
    beta : float
        scale of sphericity
        
    Returns
    -------
    omega : float
        frequency of Rossby wave
    """
    omega = -beta * wavevector[0] / (wavevector[0]**2 + wavevector[1]**2)
    return omega


class RossbyWave:
    """
    Class to represent a Rossby wave.

    Attributes
    ----------
    wavevector : array_like
        wavevector (k, l) of wavenumbers
    k : float
        1st component of wavevector
    l : float
        2nd component of wavevector
    phase : float
        phase of the wave
    beta : float
        scale of sphericity

    Methods
    -------
    __str__(self):
        Return string representation: RossbyWave([k, l], phase).
    __repr__(self):
        Return canonical string representation: RossbyWave([k, l], phase, beta).
    streamfunction(x, t):
        Return streamfunction of Rossby wave.
    plot_streamfunction(self, xlim=(-1, 1, 100), ylim=(-1, 1, 100), t=0, lines=50, filled=True):
        Contour plot of the streamfunction of a Rossby wave.
    animate_streamfunction(self, xlim=(-1, 1, 100), ylim=(-1, 1, 100), tlim=(0, 1000, 101), lines=50, filled=True, filename="streamfunction"):
        Animate the contour plot of the streamfunction of a Rossby wave.
    potentialfunction(self, x, y, t, eps=0.1):
        Return streamfunction of Rossby wave.
    velocity(self, x, y, t, eps=0.1, irrotational=False, solenoidal=False):
        Return velocity of Rossby wave at x at time t.
    plot_velocity(self, xlim=(-1, 1, 100), ylim=(-1, 1, 100), t=0, density=1, eps=0.1, irrotational=False, solenoidal=False):
        Quiverplot the velocity of the Rossby wave.
    plot_stream_velocity(self, xlim=(-np.pi, np.pi, 20, 100), ylim=(-np.pi, np.pi, 20, 100), t=0, eps=0.1, irrotational=False, solenoidal=False, lines=50):
        Contourplot of streamfunction with quiverplot of velocity field of a Rossby wave.
    animate_velocity(self, xlim=(-np.pi, np.pi, 20), ylim=(-np.pi, np.pi, 20), tlim=(0, 3e13, 100), eps=0.1, irrotational=False, solenoidal=False, filename="velocityfield"):
        Animate the quiver plot of the velocity field of a Rossby wave.
    velocity_divergence(self, x, y, t, eps=0.1):
        Return the velocity divergence at (x, y) at time t.
    plot_velocity_divergence(self, xlim=(-np.pi, np.pi, 100), ylim=(-np.pi, np.pi, 100), t=0, eps=0.1, lines=50):
        Contour plot of the streamfunction of a Rossby wave.
    animate_velocity_divergence(self, xlim=(-np.pi, np.pi, 100), ylim=(-np.pi, np.pi, 100), tlim=(0, 3e13, 100), eps=0.1, lines=50, filename="velocity_divergence"):
        Animate the contour plot of the streamfunction of a Rossby wave.
    animate_trajectory(self, x0, xlim=(-np.pi, np.pi, 20, 100), ylim=(-np.pi, np.pi, 20, 100), tlim=(0, 3e13, 100), lines=50, markersize=2, eps=0.1, irrotational=False, solenoidal=False, filename="trajectory"):
        Animate the quiver plot of the velocity field of a Rossby wave.
    """

    def __init__(self, wavevector, phase=0, beta=beta):
        self.wavevector = list(wavevector)
        self.k = wavevector[0]
        self.l = wavevector[1]
        self.phase = phase
        self.beta = beta
        self.omega = -beta * wavevector[0] / (wavevector[0]**2 + wavevector[1]**2)
        self.amplitude = np.exp(-wavevector[0]**2 / 25 - wavevector[1]**2 / 25) * (
                         wavevector[0]**2 + wavevector[1]**2)

    def __str__(self):
        """Return string representation: RossbyWave([k, l], phase)."""
        return self.__class__.__name__ + "(" + str(
            self.wavevector) + ", " + str(self.phase) + ")"

    def __repr__(self):
        """Return canonical string representation: RossbyWave([k, l], phase, beta)."""
        return self.__class__.__name__ + "(" + repr(
            self.wavevector) + ", " + repr(self.phase) + ", " + repr(
                self.beta) + ")"

    def streamfunction(self, x, y, t):
        """
        Return streamfunction of Rossby wave.

        Parameters
        ----------
        x : float
            x position coordinate
        y : float
            y position coordinate
        t : float
            time

        Returns
        -------
        psi : float
            streamfunction at x at time t
        """
        psi = amplitude(self.wavevector) * np.cos(
            self.k * x + self.l * y -
            dispersion(self.wavevector, self.beta) * t + self.phase)
        return psi

    def plot_streamfunction(self,
                            xlim=(-np.pi, np.pi, 100),
                            ylim=(-np.pi, np.pi, 100),
                            t=0,
                            lines=50,
                            filled=True):
        """
        Contour plot of the streamfunction of a Rossby wave.

        Parameters
        ----------
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        t : float
            time
        lines : float
            scale of number of lines
        filled : bool
            if false, plot contour and if true, plot filled contour

        Returns
        -------
        """
        # assumes t units are 0.1 microseconds
        time = round(t / (10_000_000 * 60 * 60 * 24), 1)
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        X, Y = np.meshgrid(x, y)
        Z = self.streamfunction(X, Y, t)
        if filled:
            plt.contourf(X, Y, Z, lines, cmap="coolwarm")
        else:
            plt.contour(X, Y, Z, lines, cmap="coolwarm")
        plt.xlabel('X')
        plt.ylabel('Y')
        cbar = plt.colorbar(pad=0.1)
        cbar.ax.set_ylabel("Stream Function value")
        plt.title(f"t={time} days")

    def animate_streamfunction(self,
                               xlim=(-np.pi, np.pi, 100),
                               ylim=(-np.pi, np.pi, 100),
                               tlim=(0, 3e13, 100),
                               lines=50,
                               filled=True,
                               filename="streamfunction"):
        """
        Animate the contour plot of the streamfunction of a Rossby wave.

        Parameters
        ----------
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        tlim : array_like
            (time start, time end, time points)
        lines : float
            scale of number of lines
        filled : bool
            if false, plot contour and if true, plot filled contour
        filename : str
            file saved as {filename}.gif

        Returns
        -------
        """
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        t = np.linspace(*tlim)
        xx, yy = np.meshgrid(x, y)
        Y, T, X = np.meshgrid(y, t, x)
        fig, ax = plt.subplots(1)
        stream = self.streamfunction(X, Y, T)

        def init_func():
            plt.cla()

        def update_plot(i):
            plt.cla()
            plt.xlabel('X')
            plt.ylabel('Y')
            if not isinstance(self, RossbyOcean):
                plt.title(
                    f"RossbyWave: k={self.k}, l={self.l}, phase={self.phase}")
            else:
                plt.title("RossbyOcean")
            if filled:
                plt.contourf(xx, yy, stream[i], lines, cmap="coolwarm")
            else:
                plt.contour(xx, yy, stream[i], lines, cmap="coolwarm")

        anim = FuncAnimation(fig,
                             update_plot,
                             frames=np.arange(0, len(t)),
                             init_func=init_func)

        writergif = PillowWriter(fps=30)
        anim.save(f'{filename}.gif', writer=writergif)

    def potentialfunction(self, x, y, t, eps=0.1):
        """
        Return streamfunction of Rossby wave.

        Parameters
        ----------
        x : float
            x position coordinate
        y : float
            y position coordinate
        t : float
            time
        eps : float
            ratio of stream to potential function

        Returns
        -------
        phi : float
            potentialfunction at x at time t
        """
        phi = eps * self.streamfunction(x, y, t)
        return phi

    def velocity(self, x, y, t, eps=0.1, irrotational=False, solenoidal=False):
        """
        Return velocity of Rossby wave at x at time t.

        Parameters:
        x : float
            x position coordinate
        y : float
            y position coordinate
        t : float
            time
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave

        Returns
        -------
        v : np.ndarray
            velocity at x at time t
        """
        # v = (-dpsi/dy, dpsi/dx) + (dphi/dx, dphi/dy)
        # psi = 1/eps * phi = Re[A * exp[i(kx + ly - omega * t + phase)]]

        # dpsi/dx = Re[A * ik * exp[i(kx + ly - omega * t + phase)]]
        #         = A * -k * sin(kx + ly - omega * t + phase)
        # dpsi/dy = Re[A * il * exp[i(kx + ly - omega * t + phase)]]
        #         = A * -l * sin(kx + ly - omega * t + phase)

        v = [0, 0]
        if irrotational and solenoidal:
            raise ValueError(
                "Wave cannot be both irrotational and solenoidal.")
        if not irrotational:
            # no phi
            v[0] += amplitude(self.wavevector) * self.l * np.sin(
                self.k * x + self.l * y -
                dispersion(self.wavevector, self.beta) * t + self.phase)
            v[1] += -amplitude(self.wavevector) * self.k * np.sin(
                self.k * x + self.l * y -
                dispersion(self.wavevector, self.beta) * t + self.phase)
        if not solenoidal:
            # no psi
            v[0] += -eps * amplitude(self.wavevector) * self.k * np.sin(
                self.k * x + self.l * y -
                dispersion(self.wavevector, self.beta) * t + self.phase)
            v[1] += -eps * amplitude(self.wavevector) * self.l * np.sin(
                self.k * x + self.l * y -
                dispersion(self.wavevector, self.beta) * t + self.phase)
        return np.array(v)

    def plot_velocity(self,
                      xlim=(-np.pi, np.pi, 20),
                      ylim=(-np.pi, np.pi, 20),
                      t=0,
                      eps=0.1,
                      irrotational=False,
                      solenoidal=False):
        """
        Quiverplot the velocity of the Rossby wave.
        
        Parameters
        ----------
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        t : float
            time
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave
            
        Returns
        -------
        """
        time = round(t / (10_000_000 * 60 * 60 * 24), 1)
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        X, Y = np.meshgrid(x, y)
        u, v = self.velocity(X,
                             Y,
                             t,
                             eps=eps,
                             irrotational=irrotational,
                             solenoidal=solenoidal)
        plt.quiver(X, Y, u, v)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"t={time} days")

    def animate_velocity(self,
                         xlim=(-np.pi, np.pi, 20),
                         ylim=(-np.pi, np.pi, 20),
                         tlim=(0, 3e13, 100),
                         eps=0.1,
                         irrotational=False,
                         solenoidal=False,
                         filename="velocityfield"):
        """
        Animate the quiver plot of the velocity field of a Rossby wave.

        Parameters
        ----------
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        tlim : array_like
            (time start, time end, time points)
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave
        filename : str
            file saved as {filename}.gif

        Returns
        -------
        """
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        t = np.linspace(*tlim)
        xx, yy = np.meshgrid(x, y)
        Y, T, X = np.meshgrid(y, t, x)
        fig, ax = plt.subplots(1)
        u, v = self.velocity(X,
                             Y,
                             T,
                             eps=eps,
                             irrotational=irrotational,
                             solenoidal=solenoidal)

        def init_func():
            plt.cla()

        def update_plot(i):
            plt.cla()
            plt.xlabel('X')
            plt.ylabel('Y')
            if not isinstance(self, RossbyOcean):
                plt.title(
                    f"RossbyWave: k={self.k}, l={self.l}, phase={self.phase}")
            else:
                plt.title("RossbyOcean Velocity Field")
            plt.quiver(xx, yy, u[i], v[i])

        anim = FuncAnimation(fig,
                             update_plot,
                             frames=np.arange(0, len(t)),
                             init_func=init_func)

        writergif = PillowWriter(fps=30)
        anim.save(f'{filename}.gif', writer=writergif)

    def plot_stream_velocity(self,
                             xlim=(-np.pi, np.pi, 20, 100),
                             ylim=(-np.pi, np.pi, 20, 100),
                             t=0,
                             eps=0.1,
                             irrotational=False,
                             solenoidal=False,
                             lines=50):
        """
        Contourplot of streamfunction with quiverplot of velocity field of a Rossby wave.

        Parameters
        ----------
        t : float
            time
        xlim : array_like
            (x start, x end, x velocity points, x stream points)
        ylim : array_like
            (y start, y end, y velocity points, y stream points)
        density : float
            density of streamplot arrows
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave
        lines : float
            scale of number of lines
            
        Returns
        -------
        """
        time = round(t / (10_000_000 * 60 * 60 * 24), 1)
        xlim, ylim = list(xlim), list(ylim)
        x_vel, y_vel = np.linspace(*xlim[0:-1]), np.linspace(*ylim[0:-1])
        xlim.pop(2)
        ylim.pop(2)
        x_str, y_str = np.linspace(*xlim), np.linspace(*ylim)

        # contour plot
        X_str, Y_str = np.meshgrid(x_str, y_str)
        Z_str = self.streamfunction(X_str, Y_str, t)
        cplot = plt.contourf(X_str, Y_str, Z_str, lines, cmap="coolwarm")
        cbar = plt.colorbar(cplot)
        cbar.ax.set_ylabel("Stream Function value")

        # quiver plot
        X_vel, Y_vel = np.meshgrid(x_vel, y_vel)
        u_vel, v_vel = self.velocity(X_vel,
                                     Y_vel,
                                     t,
                                     eps=eps,
                                     irrotational=irrotational,
                                     solenoidal=solenoidal)
        quiv = plt.quiver(X_vel, Y_vel, u_vel, v_vel)

        # labels
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"t={time} days, eps={eps}")

    def animate_stream_velocity(self,
                                xlim=(-np.pi, np.pi, 20, 100),
                                ylim=(-np.pi, np.pi, 20, 100),
                                tlim=(0, 3e13, 100),
                                lines=50,
                                eps=0.1,
                                irrotational=False,
                                solenoidal=False,
                                filename="streamvelocityfield"):
        """
        Animate the quiver plot of the velocity field of a Rossby wave.

        Parameters
        ----------
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        tlim : array_like
            (time start, time end, time points)
        lines : float
            scale of number of lines
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave
        filename : str
            file saved as {filename}.gif

        Returns
        -------
        """
        xlim, ylim = list(xlim), list(ylim)
        x_vel, y_vel = np.linspace(*xlim[0:-1]), np.linspace(*ylim[0:-1])
        xlim.pop(2)
        ylim.pop(2)
        x_str, y_str = np.linspace(*xlim), np.linspace(*ylim)
        t = np.linspace(*tlim)

        xx_vel, yy_vel = np.meshgrid(x_vel, y_vel)
        Y_vel, T_vel, X_vel = np.meshgrid(y_vel, t, x_vel)
        xx_str, yy_str = np.meshgrid(x_str, y_str)
        Y_str, T_str, X_str = np.meshgrid(y_str, t, x_str)
        fig, ax = plt.subplots(1)
        u, v = self.velocity(X_vel,
                             Y_vel,
                             T_vel,
                             eps=eps,
                             irrotational=irrotational,
                             solenoidal=solenoidal)
        stream = self.streamfunction(X_str, Y_str, T_str)

        def init_func():
            plt.cla()

        def update_plot(i):
            plt.cla()
            plt.xlabel('X')
            plt.ylabel('Y')
            if not isinstance(self, RossbyOcean):
                plt.title(
                    f"RossbyWave: k={self.k}, l={self.l}, phase={self.phase}")
            else:
                plt.title("RossbyOcean Streamfunction on Velocity Field")
            plt.contourf(xx_str, yy_str, stream[i], lines, cmap="coolwarm")
            plt.quiver(xx_vel, yy_vel, u[i], v[i])

        anim = FuncAnimation(fig,
                             update_plot,
                             frames=np.arange(0, len(t)),
                             init_func=init_func)

        writergif = PillowWriter(fps=30)
        anim.save(f'{filename}.gif', writer=writergif)

    def velocity_divergence(self, x, y, t, eps=0.1):
        """
        Return the velocity divergence at (x, y) at time t.

        Parameters
        ----------
        x : float
            x coordinate
        y : float
            y coordinate
        t : float
            time
        eps : float
            ratio of stream to potential function

        Returns
        -------
        div : float
            divergence of velocity potential
        """
        # v = (-dpsi/dy, dpsi/dx) + (dphi/dx, dphi/dy)
        # psi = 1/eps * phi = Re[A * exp[i(kx + ly - omega * t + phase)]]
        # div(v) = d^2phi/dx^2 + d^2phi/dy^2
        #        = eps * [Re(d/dx A * ik * exp[i(kx + ly - omega * t + phase)] + d/dy A * il * exp[i(kx + ly - omega * t + phase)])]
        #        = eps * [Re[(A * -k^2 + A * -l^2)exp[i(kx + ly - omega * t + phase)]]
        #        = eps * A * (-l^2-k^2) * cos(kx + ly - omega * t + phase)

        div = eps * amplitude(
            self.wavevector) * (-self.l**2 - self.k**2) * np.cos(
                self.k * x + self.l * y -
                dispersion(self.wavevector, self.beta) * t + self.phase)
        return div

    def plot_velocity_divergence(self,
                                 xlim=(-np.pi, np.pi, 100),
                                 ylim=(-np.pi, np.pi, 100),
                                 t=0,
                                 eps=0.1,
                                 lines=50):
        """
        Contour plot of the streamfunction of a Rossby wave.

        Parameters
        ----------
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        t : float
            time
        eps : float
            ratio of stream to potential function
        lines : float
            scale of number of lines

        Returns
        -------
        """
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        X, Y = np.meshgrid(x, y)
        Z = self.velocity_divergence(X, Y, t, eps)
        plt.contourf(X, Y, Z, lines, cmap="coolwarm")
        plt.xlabel('X')
        plt.ylabel('Y')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("Divergence value")
        if not isinstance(self, RossbyOcean):
            plt.title(
                f"RossbyWave: k={self.k}, l={self.l}, phase={self.phase}")
        else:
            plt.title("RossbyOcean Divergence")

    def animate_velocity_divergence(self,
                                    xlim=(-np.pi, np.pi, 100),
                                    ylim=(-np.pi, np.pi, 100),
                                    tlim=(0, 3e13, 100),
                                    eps=0.1,
                                    lines=50,
                                    filename="velocity_divergence"):
        """
        Animate the contour plot of the streamfunction of a Rossby wave.

        Parameters
        ----------
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        tlim : array_like
            (time start, time end, time points)
        eps : float
            ratio of stream to potential function
        lines : float
            scale of number of lines
        filename : str
            file saved as {filename}.gif

        Returns
        -------
        """
        x = np.linspace(*xlim)
        y = np.linspace(*ylim)
        t = np.linspace(*tlim)
        xx, yy = np.meshgrid(x, y)
        Y, T, X = np.meshgrid(y, t, x)
        fig, ax = plt.subplots(1)
        div = self.velocity_divergence(X, Y, T)

        def init_func():
            plt.cla()

        def update_plot(i):
            plt.cla()
            plt.xlabel('X')
            plt.ylabel('Y')
            if not isinstance(self, RossbyOcean):
                plt.title(
                    f"RossbyWave: k={self.k}, l={self.l}, phase={self.phase}")
            else:
                plt.title("RossbyOcean")
            plt.contourf(xx, yy, div[i], lines, cmap="coolwarm")

        anim = FuncAnimation(fig,
                             update_plot,
                             frames=np.arange(0, len(t)),
                             init_func=init_func)

        writergif = PillowWriter(fps=30)
        anim.save(f'{filename}.gif', writer=writergif)

    def animate_trajectory(self,
                           x0,
                           xlim=(-np.pi, np.pi, 20, 100),
                           ylim=(-np.pi, np.pi, 20, 100),
                           tlim=(0, 10, 100),
                           lines=50,
                           markersize=2,
                           eps=0.1,
                           irrotational=False,
                           solenoidal=False,
                           filename="trajectory"):
        """
        Animate the quiver plot of the velocity field of a Rossby wave.

        Parameters
        ----------
        x0 : np.array
            initial position of particle
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        tlim : array_like
            (time start, time end, time points)
        lines : float
            scale of number of lines
        markersize : float
            size of markers plotting trajectory
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave
        filename : str
            file saved as {filename}.gif

        Returns
        -------
        """
        x, y = trajectory(self, x0, tlim[0], tlim[1], tlim[2] * 50, eps,
                          irrotational, solenoidal)

        xlim, ylim = list(xlim), list(ylim)
        x_vel, y_vel = np.linspace(*xlim[0:-1]), np.linspace(*ylim[0:-1])
        xlim.pop(2)
        ylim.pop(2)
        x_str, y_str = np.linspace(*xlim), np.linspace(*ylim)
        t = np.linspace(*tlim)

        xx_vel, yy_vel = np.meshgrid(x_vel, y_vel)
        Y_vel, T_vel, X_vel = np.meshgrid(y_vel, t, x_vel)
        xx_str, yy_str = np.meshgrid(x_str, y_str)
        Y_str, T_str, X_str = np.meshgrid(y_str, t, x_str)
        fig, ax = plt.subplots(1)
        u, v = self.velocity(X_vel,
                             Y_vel,
                             T_vel,
                             eps=eps,
                             irrotational=irrotational,
                             solenoidal=solenoidal)
        stream = self.streamfunction(X_str, Y_str, T_str)

        def init_func():
            plt.cla()

        def update_plot(i):
            x_traj = x[0:i * 50]
            y_traj = y[0:i * 50]
            plt.cla()
            plt.xlabel('X')
            plt.ylabel('Y')
            if not isinstance(self, RossbyOcean):
                plt.title(
                    f"RossbyWave: k={self.k}, l={self.l}, phase={self.phase}")
            else:
                plt.title("RossbyOcean Streamfunction on Velocity Field")
            plt.contourf(xx_str, yy_str, stream[i], lines, cmap="coolwarm")
            plt.quiver(xx_vel, yy_vel, u[i], v[i])
            plt.plot(x_traj, y_traj, 'o', ms=markersize, color="magenta")

        anim = FuncAnimation(fig,
                             update_plot,
                             frames=np.arange(0, len(t)),
                             init_func=init_func)

        writergif = PillowWriter(fps=30)
        anim.save(f'{filename}.gif', writer=writergif)


class RossbyOcean(RossbyWave):
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
    __str__(self):
        Return string representation: RossbyOcean(RossbyWave(wavevector, phase), ...).
    __repr__(self):
        Return canonical string representation: RossbyOcean([RossbyWave(wavevector, phase, beta), ...], beta).
    streamfunction(x, t):
        Return streamfunction of Rossby wave.
    velocity(self, x, y, t, eps=0.1, irrotational=False, solenoidal=False):
        Return velocity of Rossby wave at x at time t.
    velocity_divergence(self, x, y, t, eps=0.1):
        Return the velocity divergence at (x, y) at time t.
    add_wave(wave):
        Add a RossbyWave to the RossbyOcean.
    add_random_wave(self, xlim=(-5, 5), ylim=(-5, 5), plim=(0, 2 * np.pi)):
        Add a RossbyWave to the Rossbyocean with random wavevector.
    add_random_waves(self, n, xlim=(-5, 5), ylim=(-5, 5), plim=(0, 2 * np.pi)):
        Add n random wavevectors.
    normal_wavevectors(xlim=(-5, 5, 10), ylim=(-5, 5, 10)):
        Add RossbyWaves with wavevectors (k, l) in a grid.
    remove_wave(self, index):
        Remove the RossbyWave at index in the RossbyOcean.
    """

    def __init__(self, rossby_waves, beta=beta):
        self.waves = rossby_waves
        self.wavevectors = np.array([wave.wavevector for wave in rossby_waves])
        self.phases = np.array([wave.phase for wave in rossby_waves])
        self.k = self.wavevectors[:, 0]
        self.l = self.wavevectors[:, 1]
        self.beta = beta

    def __str__(self):
        """
        Return string representation:
        RossbyOcean(RossbyWave(wavevector, phase), ...).
        """
        waves = ""
        for wave in self.waves:
            waves += str(wave) + ", "
        return self.__class__.__name__ + "(" + waves[0:-2] + ")"

    def __repr__(self):
        """
        Return canonical string representation:
        RossbyOcean([RossbyWave(wavevector, phase, beta), ...], beta).
        """
        return self.__class__.__name__ + "(" + str(self.waves) + ", " + str(
            self.beta) + ")"

    def streamfunction(self, x, y, t):
        """
        Return streamfunction of Rossby ocean.

        Parameters
        ----------
        x : float
            x position coordinate
        y : float
            y position coordinate
        t : float
            time

        Returns
        -------
        psi : float
            streamfunction at x at time t
        """
        psi = 0
        for wave in self.waves:
            psi += wave.streamfunction(x, y, t)
        return psi

    def velocity(self, x, y, t, eps=0.1, irrotational=False, solenoidal=False):
        """
        Return velocity of Rossby wave at x at time t.

        Parameters:
        x : float
            x position coordinate
        y : float
            y position coordinate
        t : float
            time
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave

        Returns
        -------
        v : np.ndarray
            velocity at x at time t
        """
        # v = (-dpsi/dy, dpsi/dx) + (dphi/dx, dphi/dy)
        # eps*phi = psi = A * exp(k.x - omega * t)
        v = [0, 0]
        if irrotational and solenoidal:
            raise ValueError(
                "Wave cannot be both irrotational and solenoidal.")
        for wave in self.waves:
            ou, ov = wave.velocity(x, y, t, eps, irrotational, solenoidal)
            v[0] += ou
            v[1] += ov
        return np.array(v)

    def velocity_divergence(self, x, y, t, eps=0.1):
        """
        Return the velocity divergence at (x, y) at time t.
        
        Parameters
        ----------
        x : float
            x coordinate
        y : float
            y coordinate
        t : float
            time
        eps : float
            ratio of stream to potential function

        Returns
        -------
        div : float
            divergence of velocity potential
        """
        div = 0
        for wave in self.waves:
            div += wave.velocity_divergence(x, y, t, eps)
        return div

    def add_wave(self, wave):
        """
        Add a RossbyWave to the RossbyOcean.
        
        Parameters
        ----------
        wave : RossbyWave
            RossbyWave to be added
                
        Returns
        -------
        """
        self.waves.append(wave)
        self.wavevectors = np.array([wave.wavevector for wave in self.waves])
        self.phases = np.array([wave.phase for wave in self.waves])
        self.k = self.wavevectors[:, 0]
        self.l = self.wavevectors[:, 1]
        self = RossbyOcean(self.waves, beta=self.beta)

    def remove_wave(self, index):
        """
        Remove the RossbyWave at index in the RossbyOcean.
        
        Parameters
        ----------
        index : int
            index of RossbyWave to be removed
            
        Returns
        -------
        """
        self.waves.pop(index)
        self.wavevectors = np.array([wave.wavevector for wave in self.waves])
        self.phases = np.array([wave.phase for wave in self.waves])
        self.k = self.wavevectors[:, 0]
        self.l = self.wavevectors[:, 1]
        self = RossbyOcean(self.waves, beta=self.beta)

    def add_random_wave(self,
                        xlim=(-5, 5),
                        ylim=(-5, 5),
                        plim=(0, 2 * np.pi),
                        beta=beta):
        """
        Add a RossbyWave to the Rossbyocean with random wavevector.

        Parameters
        ----------
        xlim : array_like
            (l, u) lower and upperbounds of k wavevector component
        ylim : array_like
            (l, u) lower and upperbounds of l wavevector component
        plim : array_like
            (l, u) lower and upperbounds of phase
        beta : float
            scale of sphericity

        Returns
        -------
        """
        k = 0
        l = 0
        while k == 0 and l == 0:
            k = (xlim[1] - xlim[0]) * np.random.random() + xlim[0]
            l = (ylim[1] - ylim[0]) * np.random.random() + ylim[0]
        phase = (plim[1] - plim[0]) * np.random.random() + plim[0]
        self.add_wave(RossbyWave([k, l], phase, beta=beta))

    def add_random_waves(self,
                         n,
                         beta=beta,
                         xlim=(-5, 5),
                         ylim=(-5, 5),
                         plim=(0, 2 * np.pi)):
        """
        Add n random wavevectors.
        
        Parameters
        ----------
        n : int
            number of wavevectors to add
        beta : float
            scale of sphericity
        xlim : array_like
            (l, u) lower and upperbounds of k wavevector component
        ylim : array_like
            (l, u) lower and upperbounds of l wavevector component
        plim : array_like
            (l, u) lower and upperbounds of phase

        Returns
        -------
        """
        for i in range(n):
            self.add_random_wave(xlim, ylim, plim, beta)

    def add_grid_waves(self, xlim=(-5, 5, 11), ylim=(-5, 5, 11), phase=True):
        """
        Add RossbyWaves with wavevectors (k, l) in a grid.
        
        Parameters
        ----------
        xlim : array_like
            (x start, x end, x points)
        ylim : array_like
            (y start, y end, y points)
        phase : bool
            if True, add random phases in (0, 2*np.pi), else phase=0
        
        Returns
        -------
        """
        x, y = np.linspace(*xlim), np.linspace(*ylim)
        for i in x:
            for j in y:
                if i == 0 and j == 0:
                    continue
                if phase:
                    p = 2 * np.pi * np.random.random()
                else:
                    p = 0
                self.add_wave(RossbyWave((i, j), p, beta=2e-11))


def trajectory(ro, x0, t0, t, h, eps=0.1, irrotational=False, solenoidal=False, xrange=np.pi, yrange=np.pi, constant=False):
    """
        Return lists of x-coords and y-coords of trajectory of particles with
        initial conditions in the velocity field of the RossbyOcean.

        Parameters
        ----------
        ro : RossbyOcean

        x0 : nx2 np.array
            initial positions of n particles
        t0 : float
            starting time
        t : float
            ending time
        h : float
            step size
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave
        xrange : float
            length of x-axis, e.g. if xrange = 3, the x-axis goes from -3 to 3
        yrange : float
            length of y-axis, e.g. if yrange = 3, the y-axis goes from -3 to 3

        Returns
        -------
        x_coords : list
            list of lists of x-coordinates of particle trajectories, e.g. x[i] gives x-coordinates of the i-1th particle's trajectory
        y_coords : list
            list of lists of y-coordinates of particle trajectories, e.g. y[i] gives y-coordinates of the i-1th particle's trajectory
        """
    n = (t - t0) / h
    x = x0
    t = t0
    i = 0
    num_points = x0.shape[0]
    x_coords = [[] for x in range(num_points)]
    y_coords = [[] for x in range(num_points)]
    j = 0
    for list in x_coords:
        list.append(x[j, 0])
        j += 1
    j = 0
    for list in y_coords:
        list.append(x[j, 1])
        j += 1
    while i < n:
        k_1 = vel(ro, x, t, eps, irrotational, solenoidal)
        k_2 = vel(ro, x + h * k_1 / 2, t + h / 2, eps, irrotational, solenoidal)
        k_3 = vel(ro, x + h * k_2 / 2, t + h / 2, eps, irrotational, solenoidal)
        k_4 = vel(ro, x + h * k_3, t + h, eps, irrotational, solenoidal)
        x = x + h / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        i += 1
        t += h
        x[:, 0] += xrange
        x[:, 0] = (x[:, 0] % (2 * xrange)) - xrange
        x[:, 1] += yrange
        x[:, 1] = (x[:, 1] % (2 * yrange)) - yrange
        j = 0
        for list in x_coords:
            list.append(x[j, 0])
            j += 1
        j = 0
        for list in y_coords:
            list.append(x[j, 1])
            j += 1

    return x_coords, y_coords


def grid(n, xrange=np.pi, yrange=np.pi):
    """
        Return array of nxn evenly spaced points.
        
        Parameters
        ----------
        n : int
            
        xrange : float
            length of half of x-axis, e.g. xrange = 1 means the x-axis goes from -1 to 1
        yrange : float
            length of half of y-axis
        
        Returns
        -------
        np.array
        """
    a, b = -xrange, xrange
    h = (b-a)/n
    x = [a+h*(1/2 + i) for i in range(n)]
    a, b = -yrange, yrange
    h = (b-a)/n
    y = [a+h*(1/2 + i) for i in range(n)]
    v = []
    for i in x:
        for j in y:
            v.append([i, j])
    return np.array(v)


def vel(ro, x, t, eps=0.1, irrotational=False, solenoidal=False):
    """
        Return array of velocity at x and time t of RossbyOcean.
        
        Parameters
        ----------
        ro : RossbyOcean
        
        x : nx2 np.array
            n points at which velocity will be evaluated at
        t : float
            time at which velocity will be evaluated at
        eps : float
            ratio of stream to potential function
        Returns
        -------
        v : nx2 np.array
            velocity of the n points at t
        """
    L = np.shape(x)[0]
    v = np.zeros((L, 2))
    if eps != 1:
        for r in ro.waves:
            v[:, 0] += (1 - eps) * r.amplitude * r.l * np.sin(r.k*x[:, 0] + r.l*x[:, 1] - r.omega*t + r.phase)
            v[:, 1] -= (1 - eps) * r.amplitude * r.k * np.sin(r.k*x[:, 0] + r.l*x[:, 1] - r.omega*t + r.phase)
    if eps != 0:
        for r in ro.waves:
            v[:, 0] += -eps * r.amplitude * r.k * np.sin(r.k*x[:, 0] + r.l*x[:, 1] - r.omega*t + r.phase)
            v[:, 1] += -eps * r.amplitude * r.l * np.sin(r.k*x[:, 0] + r.l*x[:, 1] - r.omega*t + r.phase)
    return v


def trajectory2(r, x0, t0, t, h, eps=0.1, irrotational=False, solenoidal=False, xrange=np.pi, yrange=np.pi, constant=False):
    """
        Return lists of x-coords and y-coords of trajectory of particle with
        initial conditions in the velocity field of the RossbyWave/Ocean.

        Parameters
        ----------
        r : RossbyWave or RossbyOcean
            (x start, x end, x points)
        x0 : nx2 np.array
            initial positions of particles
        t0 : float
            starting time
        n : int
            number of timesteps
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave
        xrange : float
            length of x-axis, e.g. if xrange = 3, the x-axis goes from -3 to 3
        yrange : float
            length of y-axis, e.g. if yrange = 3, the y-axis goes from -3 to 3

        Returns
        -------
        x_coords : list
            list of x-coordinates of particle trajectory
        y_coords : list
            list of y-coordinates of particle trajectory
        times : list
            list of times of particle trajectory
        """
    f = rossby_velocity(r, eps, irrotational, solenoidal)
    n = (t - t0) / h
    x = x0
    t = t0
    i = 0
    trajectory = [x]

    while i < n:
        k_1 = f(x, t)
        k_2 = f(x + h * k_1 / 2, t + h / 2)
        k_3 = f(x + h * k_2 / 2, t + h / 2)
        k_4 = f(x + h * k_3, t + h)
        x = x + h / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        i += 1
        t += h
        x[0] += xrange
        x[0] = (x[0] % (2 * xrange)) - xrange
        x[1] += yrange
        x[1] = (x[1] % (2 * yrange)) - yrange
        trajectory.append(x)
    x_coords = [x[0] for x in trajectory]
    y_coords = [x[1] for x in trajectory]
    return x_coords, y_coords
