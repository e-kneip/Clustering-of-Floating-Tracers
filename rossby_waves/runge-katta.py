def rossby_velocity(r, eps=0.1, irrotational=False, solenoidal=False):
    """
        Take RossbyWave or RossbyOcean and return the velocity field function.

        Parameters
        ----------
        r : RossbyWave or RossbyOcean
        eps : float
            ratio of stream to potential function
        irrotational : bool
            curl-free wave
        solenoidal : bool
            divergence-free wave

        Returns
        -------
        f : function
            velocity function which takes position vector, time and returns
            velocity vector
        """
    def f(x, t):
        return r.velocity(x[0], x[1], t, eps, irrotational, solenoidal)
    return f


def trajectory(r, x0, t0, t, n, eps=0.1, irrotational=False,
               solenoidal=False):
    """
        Return lists of x-coords and y-coords of trajectory of particle with
        initial conditions in the velocity field of the RossbyWave/Ocean.

        Parameters
        ----------
        r : RossbyWave or RossbyOcean
            (x start, x end, x points)
        x0 : np.array
            initial position of particle
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

        Returns
        -------
        f : function
            velocity function which takes position vector and time and returns
            velocity vector
        """
    f = rossby_velocity(r, eps, irrotational, solenoidal)
    h = t/n
    x = x0
    t = t0
    i = 0
    trajectory = [x]
    while i < n:
        k_1 = f(x, t)
        k_2 = f(x + h*k_1/2, t + h/2)
        k_3 = f(x + h*k_2/2, t + h/2)
        k_4 = f(x + h*k_3, t + h)
        x = x + h/6*(k_1 + 2*k_2 + 2*k_3 + k_4)
        i += 1
        t += h
        trajectory.append(x)
    x_coords = [x[0] for x in trajectory]
    y_coords = [x[1] for x in trajectory]
    return x_coords, y_coords

