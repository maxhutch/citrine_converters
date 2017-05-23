from __future__ import division

import numpy as np


class HoughSpace(np.ndarray):
    __doc__ = r"""
    Constructs a Hough transform space of the `xdata` and
    `ydata`.

    Construct a Hough space that, given an orientation,
    determines the distance to a point.

    For $ax + by + c = 0$, $a = \sin \phi$ and $b = \cos \phi$,\*

    $$
        d = \frac{|a x_0 + b y_0 + c|}{\sqrt{a^2 + b^2}}
    $$

    Then the distance from a line oriented at $\phi$ at
    the origin to a point $(x_0, y_0)$ is

    $$
        d = |x_0 \sin \phi + y_0 \cos \phi|
    $$

    *Note* The Hough transform can be performed on the scaled
    data, not the original data, because of the extreme
    compression of the Hough space that is a consequence of
    the highly disparate x and y axes.

    \* This can be cast into a more familiar form:

    $$
        \begin{align*}
        ax + by + c &= 0 \\
        y &= -\frac{a}{b} x - \frac{c}{b} \\
          &= -\frac{\sin (\phi)}{\cos (\phi)} x - \frac{c}{\cos (\phi)} \\
          &= \frac{\sin (-\phi)}{\cos (-\phi)} x - \frac{c}{\cos (-\phi)} \\
          &= \tan (\pi-\phi) x + \frac{c}{\cos(\pi - \phi)} \\
          &= \tan (\theta) x + \frac{c}{\cos(\theta)} \\
        y &= m x + b
        \end{align*}
    $$

    $+\phi$ is counterclockwise.

    Input
    =====
    :xdata, array-like: x data
    :ydata, array-like: y data

    Options
    =======
    :nq, int (optional): number of theta divisions.
        Default: 1801.
    :nq, int (optional): number of radial divisions.
        Default: 1801.
    """

    @staticmethod
    def distance(x, y, phi):
        """
        Shortest distance between the origin and the line that
        forms an angle $\phi$ with the x-axis and passes through
        the point (x,y).

        IN
        ==
        :x, float or ndarray: x coordinate(s)
        :y, float or ndarray: y coordinate(s)
        :phi, float or ndarray: angle(s) of the line(s)
            that pass through (x,y).
        """
        return np.abs(x*np.sin(phi) + y*np.cos(phi))

    def __new__(cls, xdata, ydata, **kwds):
        # handle options
        nq = kwds.get('nq', 1801)
        nr = kwds.get('nr', 1801)
        # set number of theta divisions
        try:
            nq = int(nq)
        except ValueError:
            msg = 'The number of theta divisions must be an integer.'
            raise ValueError(msg)
        # set number of radial divisions
        try:
            nr = int(nr)
        except ValueError:
            msg = 'The number of radial divisions must be an integer.'
            raise ValueError(msg)
        # initialize the hough space
        obj = np.zeros((nq, nr), dtype=int).view(cls)
        obj.theta = (0, np.pi)
        obj.radius = (0, 1)
        obj.nq = nq
        obj.nr = nr
        # build conditions based on options
        if not isinstance(xdata, np.ndarray):
            # why not just use asarray? in case xdata is a subclass of
            # ndarray we don't want to construct a new ndarray view
            obj.x = np.asarray(xdata)
        else:
            obj.x = xdata
        if not isinstance(ydata, np.ndarray):
            # why not just use asarray? in case ydata is a subclass of
            # ndarray we don't want to construct a new ndarray view
            obj.y = np.asarray(ydata)
        else:
            obj.y = ydata
        # construct the hough space
        obj.construct()
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        try:
            self.nq = getattr(obj, 'nq', obj.shape[0])
        except IndexError:
            self.nq = 0
        try:
            self.nr = getattr(obj, 'nr', obj.shape[1])
        except IndexError:
            self.nr = 0
        self.x  = getattr(obj, 'x', np.array([], dtype=float))
        self.y  = getattr(obj, 'y', np.array([], dtype=float))
        return obj

    def theta_distance(self, iq, ir):
        """
        Returns the theta and distance values for a given coordinate
        in the Hough space.

        Input
        =====
        :iq, int: theta index for the point in the Hough space
        :ir, int: radius/distance index for the point in the Hough space.

        Output
        ======
        (theta, distance) as floats.
        """
        qlo, qhi = self.theta
        rlo, rhi = self.radius
        theta = iq/self.nq * (qhi - qlo) + qlo
        distance = ir/self.nr * (rhi - rlo) + rlo
        return (theta, distance)

    def construct(self):
        """
        Constructs the Hough space from the x and y point data
        stored as part of `self`.

        IN
        ==
        :self: this instance

        OUT
        ===
        None. `self.hough` is created/updated on this call.
        """
        assert self.x.shape == self.y.shape, \
            "The shapes of the x and y vectors must match."
        nq = self.nq
        nr = self.nr
        # construct the Hough space
        #+ what range of theta and r are appropriate?
        #+ at worst, the line eminating from no point will be
        #+ farther away from the origin that the distance to the
        #+ point itself.
        radius = np.linspace(0,
            np.sqrt(self.x**2 + self.y**2).max(),
            num=nr-1)
        self.radius = (radius.min(), radius.max())
        #+ since each line extends in both directions from the point
        #+ there is only need to explore 180 degrees (pi radians)
        theta = np.linspace(0,
            np.pi,
            num=nq-1)
        self.theta = (theta.min(), theta.max())
        # see the doc string for the HoughSpace class for a detailed
        # description of the role of phi. In short, the theta from
        # the theta -> phi conversion is what one would expect from
        # $y = mx + b$ where $m = \tan \theta$ for $+\theta$
        # counterclockwise.
        phi = np.pi - theta
        # with what indices do the theta values correspond?
        qlo, qhi = theta[0], theta[-1]
        iq = ((theta - qlo)/(qhi - qlo)*(nq-1)).astype(int)
        # range of the radial values
        rlo, rhi = radius[0], radius[-1]
        # populate the Hough space
        self.fill(0)
        for x,y in zip(self.x, self.y):
            # vectorized calculation of all distances. Note the
            # use of $\phi$, not $\theta$ in this equation. The
            # reason can be found in the HoughSpace doc string.
            d = (HoughSpace.distance(x, y, phi) - rlo)/(rhi - rlo)
            # To which index does each distance correspond
            ir = (d*(nr-1)).astype(int)
            self[iq, ir] += 1
#end 'class HoughSpace(object):'
