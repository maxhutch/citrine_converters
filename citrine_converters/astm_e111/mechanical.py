from __future__ import division

import numpy as np
import warnings
from bisect import bisect_left as bisect
from scipy.ndimage import gaussian_filter
from scipy.optimize import leastsq


def linear_merge(x1, y1, x2, y2):
    """
    Merge data pairs (x1, y1) and (x2, y2) over the intersection
    of their ranges (x) using a linear interpolation of each
    dataset to interpolate between unaligned values.

    No extrapolation is performed.

    Input
    =====
    :x1, array-like: abscissa coordinates of dataset 1
    :y1, array-like: ordinate coordinates of dataset 1
    :x2, array-like: abscissa coordinates of dataset 2
    :y2, array-like: ordinate coordinates of dataset 2

    OUT
    ===
    Tuple of the merged x (`xm`) and interpolated y1 (`y1m`) and y2 (`y2m`):
    `(xm, y1m, y2m)`
    """
    # ensure all values are ndarrays
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    # ##########
    # merge on x
    xmerge = np.concatenate((np.sort(x1), np.sort(x2)))
    xmerge.sort(kind='mergesort')
    # perform interpolation
    xlo = np.max((x1.min(), x2.min()))
    xhi = np.min((x1.max(), x2.max()))
    # keep only the merged x values within the intersection of
    # the data ranges
    mask = (xmerge >= xlo) & (xmerge <= xhi)
    xf = xmerge[mask]
    y1f = np.interp(xf, x1, y1)
    y2f = np.interp(xf, x2, y2)
    # return the interpolated, merged dataset
    return (xf, y1f, y2f)


class MechanicalProperties(object):
    """
    Summary
    =======

    Determines the mechanical properties given a pd.DataFrame with
    keys "strain" (`epsilon`) and "stress" (`sigma`).

    The time at which stress and strain are recorded must by synchronized, e.g.
    stress data collected at time `t` must correspond to the strain data
    collected at that same time, `t`.

    The frequency of data collection need not match. Linear interpolation is
    used to register the stress and strain data. No extrapolation is allowed,
    so the only accessible times are the intersection of the time ranges of
    strain and stress data.

    Input
    =====
    :param epsilon, pd.DataFrame: Strain data that, at a minimum,
        must contain `time` and `strain` fields.
    :param sigma, pd.DataFrame: Stress data that, at a minimum, must
        contain `time` and `stress` fields.
    """
    def __init__(self, epsilon, sigma):
        # ##########
        # merge on time
        time, strain, stress = linear_merge(
            x1=epsilon['time'].values, y1=epsilon['strain'].values,
            x2=sigma['time'].values, y2=sigma['stress'].values)
        self.time   = time
        self.strain = strain
        self.stress = stress

    @property
    def elastic_modulus(self):
        return getattr(self, '_elastic_modulus', None)

    @elastic_modulus.setter
    def elastic_modulus(self, modulus):
        self._elastic_modulus = float(modulus)

    @property
    def elastic_onset(self):
        return getattr(self, '_elastic_onset', None)

    @elastic_onset.setter
    def elastic_onset(self, onset):
        self._elastic_onset = float(onset)

    @property
    def yield_stress(self):
        # elastic modulus has not been calculated first.
        modulus = self.elastic_modulus
        onset   = self.elastic_onset
        if modulus is None:
            msg = 'The elastic modulus must be set before yield ' \
                  'strength can be calculated.'
            raise ValueError(msg)
        if onset is None:
            msg = 'The elastic onset must be set before yield ' \
                  'strength can be calculated.'
            raise ValueError(msg)
        # calculate the yield strength
        stress = modulus*(self.self.strain - onset - 0.002)
        mask   = (self.stress < stress)
        for i in reversed(range(len(mask)-1)):
            if not mask[i]:
                yield_stress = (self.stress[i] + self.stress[i+1])/2
        return yield_stress

    @property
    def strain_at_yield(self):
        # elastic modulus has not been calculated first.
        modulus = self.elastic_modulus
        onset   = self.elastic_onset
        if modulus is None:
            msg = 'The elastic modulus must be set before the ' \
                  'strain at yield can be calculated.'
            raise ValueError(msg)
        if onset is None:
            msg = 'The elastic onset must be set before the ' \
                  'strain at yield can be calculated.'
            raise ValueError(msg)
        # calculate the strain at yield
        stress = modulus*(self.self.strain - onset - 0.002)
        mask   = (self.stress < stress)
        for i in reversed(range(len(mask)-1)):
            if not mask[i]:
                strain_at_yield = (self.strain[i] + self.strain[i+1])/2
        return strain_at_yield

    @property
    def plastic_onset(self):
        return self.strain_at_yield

    @property
    def elastic_region(self):
        onset = self.elastic_onset
        if onset is None:
            msg = 'The elastic onset must be set before the ' \
                  'elastic region can be defined.'
            raise ValueError(msg)
        try:
            yield_strain = self.strain_at_yield
        except ValueError:
            msg = 'The elastic modulus must be set before the ' \
                  'elastic region can be defined.'
        strain = self.strain
        mask = (strain > onset) & (strain < yield_strain)
        return self.strain[mask]

    @property
    def plastic_region(self):
        try:
            yield_strain = self.strain_at_yield
        except ValueError:
            msg = 'The elastic modulus must be set before the ' \
                  'elastic region can be defined.'
        strain = self.strain
        mask = (strain > yield_strain)
        return self.strain[mask]

    @property
    def ultimate_stress(self):
        return self.stress.max()

    @property
    def necking_onset(self):
        i = np.where(self.stress == self.ultimate_strength)[0]
        return self.strain[i]

    @property
    def fracture_stress(self):
        return self.stress[-1]

    @property
    def total_elongation(self):
        return self.strain[-1] - self.elastic_onset

    @property
    def ductility(self):
        """
        Returns the ductility, defined as the strain at failure
        (total elongation - recoverable elastic strain).
        """
        # erecov = recoverable strain at failure
        erecov = self.fracture_stress/self.elastic_modulus
        return (self.total_elongation - erecov)

    @property
    def toughness(self):
        """Uses simple quadrature to calculate the toughness."""
        strain = self.strain
        stress = self.stress
        return np.dot((stress[1:] + stress[:-1])/2.,
                      (strain[1:] - strain[:-1]))
#end 'class MechanicalProperties(object):'


class Normalized(np.ndarray):
    def __new__(cls, vec):
        obj = np.copy(vec).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        if hasattr(obj, 'lower') and hasattr(obj, 'range'):
            self.lower = obj.lower
            self.range = obj.range
        else:
            # normalize
            lower = obj.min()
            upper = obj.max()
            self.lower = lower
            self.range = upper - lower
            # self = (self - lower)/(upper - lower)
            self -= lower
            self /= self.range

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    @property
    def unscaled(self):
        """Returns the unscaled data (initial range)"""
        return self*self.range + self.lower
#end 'class Normalizer(object):'


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


def resample(arr, **kwds):
    """
    Resample the array-like object based on intensity.

    Input
    =====
    :arr, array-like: object to be resampled

    Options
    =======
    :num, int: number of samples to use in resampling `arr`

    Output
    ======
    Resampled array
    """
    arr = np.asarray(arr)
    num = kwds.get('num', arr.size)
    # resample
    resampled = np.zeros_like(arr)
    resampled_r = resampled.ravel()
    # construct continuous distribution function
    cdf = np.cumsum(arr.ravel())
    cdf = (cdf - cdf.min())/(cdf.max() - cdf.min())
    for _ in xrange(arr.size):
        i = bisect(cdf, np.random.random())
        resampled_r[i] += 1
    return resampled


def approximate_elastic_regime_from_hough(mechprop, **kwds):
    r"""
    Construct a Hough space from the strain and stress data, then
    analyze this Hough space to approximate the slope and intercept
    of the linear elastic region of the stress-strain curve.

    Given the equation

    $$
        ax + by + c = 0,\, a = \sin \phi,\, b = \cos \phi
    $$

    from above, and

    $$
        d = |a x + b y + c|
    $$

    then $c = \pm d$ so that

    $$
        \begin{align*}
            y &= \tan(-\phi) x \pm \frac{d}{\cos(-\phi)} \\
              &= \tan(\pi - \phi) x \mp \frac{d}{\cos(\pi - \phi)} \\
            y &= \tan(\theta) x \mp \frac{d}{\cos(\theta)}
        \end{align*}
    $$

    Then for linear segments in the set `(x, y)`, with `x` and
    `y` potentially normalized, a maximum in the Hough space
    occurs at a $(\theta, r)$ corresponding to the line that
    contains colinear points. In a stress-strain curve, this
    occurs close to the origin and near 90 degrees. Intuitively,
    one would look for the elastic region in this area.

    ## Moving back from scaled to unscaled coordinates

    As mentioned above, the Hough transform is performed on the scaled
    data to eliminate data compression due to disparate axes scales.
    We must return to an unscaled data now to eliminate the dimensional
    expansion from the scaled Hough transform.

    $$
        \begin{align*}
            y_s &= x_s \tan (\theta) - d \sec (\theta) \\
            \frac{y - y_{min}}{y_{max} - y_{min}} &= \frac{x - x_{min}}{x_{max} - x_{min}} \tan(\theta) - d \sec(\theta) \\
            y &= \left( \frac{x - x_{min}}{x_{max} - x_{min}} \tan(\theta) - d \sec(\theta) \right)(y_{max} - y_{min}) + y_{min} \\
              &= x \tan(\theta)\left( \frac{y_{max} - y_{min}}{x_{max} - x_{min}} \right) - x_{min} \tan(\theta) \left( \frac{y_{max} - y_{min}}{x_{max} - x_{min}} \right) - d (y_{max} - y_{min}) \sec(\theta) + y_{min}
        \end{align*}
    $$

    From which the elastic modulus is
    $E = \tan(\theta)(y_{max} - y_{min})/(x_{max} - x_{min})$ and the yield
    strength is defined as the intersection of the stress strain curve and

    $$
        y = (x-0.002) \tan(\theta) \left( \frac{y_{max} - y_{min}}{x_{max} - x_{min}} \right) - x_{min} \tan(\theta)\left( \frac{y_{max} - y_{min}}{x_{max} - x_{min}} \right) - d (y_{max} - y_{min}) \sec(\theta) + y_{min}
    $$

    Input
    =====
    :mechprop, MechanicalProperties: Mechanical properties object from
        which the elastic properties are to be approximated.

    Options
    =======
    :lower, float: lower angle in which to look for the modulus (in degrees).
        Default: 60 degrees.
    :upper, float: upper angle in which to look for the modulus (in degrees).
        Default: 90 degrees.
    Passed through to the construction of a HoughSpace object. See
    HoughSpace for a description of these options.

    Output
    ======
    Dictionary of predicted values:
        {
            'elastic modulus' : m,
            'elastic onset'   : -b/m,
            'elastic strain'  : x[mask],
            'elastic stress'  : y[mask],
            'resampled'       : resampled,
            'hough'           : hough }
    where
    :elastic modulus, float: slope of the elastic region (Young's modulus)
    :elastic onset, float: Onset of elasticity.
    :elastic strain, array: slice of the strain vector lying inside
        the elastic region.
    :elastic stress, array: slice of the stress vector lying inside
        the elastic region.
    :resampled, 2D numpy.ndarray: resampled hough space
    :hough, HoughSpace: hough transform of stress-strain data.
    """
    # handle keywords
    qlo = kwds.get('lower', 60)
    qhi = kwds.get('upper', 90)
    # The Hough space will result in a curve that forms a "V" shape
    # near 90 degrees. If plotted with `matplotlib.imshow(...)`, this
    # "V" will point to the left, so if iterating over rows (axis 0)
    # will find the edges of the "V", then iterating over these edge
    # points for each column will identify the point of the "V".
    # construct Hough space on normalized stress and strain
    strain = Normalized(mechprop.strain)
    stress = Normalized(mechprop.stress)
    hough = HoughSpace(strain, stress, **kwds)
    # resample Hough space
    resampled = resample(hough)
    # smooth the resampled data to eliminate noise
    resampled[:] = gaussian_filter(resampled, 3)
    resampled[:] = gaussian_filter(resampled, 3)
    resampled[:] = gaussian_filter(resampled, 3)
    # look in the 60-90 degree range for the elastic region
    qlo = int(qlo/180*hough.nq)
    qhi = int(qhi/180*hough.nq)
    sub = resampled[qlo:qhi]
    pos = np.mean(np.argwhere(sub == sub.max()), axis=0) + [qlo, 0]
    theta, distance = hough.theta_distance(*pos)
    # move from scaled to unscaled coordinates (see doc string)
    x, y = strain.unscaled, stress.unscaled
    xs, ys = strain, stress
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    dy = ymax - ymin
    dx = xmax - xmin
    dydx = dy/dx
    tanq = np.tan(theta)
    secq = 1./np.cos(theta)
    # y = m*x + b
    m = tanq*dydx
    b = -xmin*m - distance*dy*secq + ymin
    # find the plastic region (lies below `y = m*(x - 0.002) + b` line)
    plastic = (y < (m*(x - 0.002) + b))
    for i in reversed(range(plastic.size)):
        if not plastic[i]:
            plastic[:i] = False
            break
    # find the compliance region, if it exists
    compliance = (strain < -b/m)
    # mask
    mask = ((~compliance) & (~plastic))
    return {
        'elastic modulus' : m,
        'elastic onset'   : -b/m,
        'elastic strain'  : x[mask],
        'elastic stress'  : y[mask],
        'resampled'       : resampled,
        'hough'           : hough }


def set_elastic(mechprop, **kwds):
    """
    Sets the elastic properties for the provided MechanicalProperties
    object.

    Input
    =====
    :mechprop, MechanicalProperties: mechanical properties object into
        which the elastic properties will be stored.

    Options
    =======
    :approximator, f(strain, stress): Function to approximate the
        "elastic onset" and "elastic modulus", returned as keys in
        a dictionary.
    :maxiter, int: maximum number of iterations. Default: 20.
    :covariance, bool: optimize on covariance. Default.
    :rsquared, bool: optimize on $R^2$. Supercedes covariance.
    :residual, float: residual strain cutoff. Default: 0.00005.

    Output
    ======
    None. Results stored in `mechprop`.
    """
    def residual_strain(strain, stress, modulus):
        return strain - modulus*stress
    def stats(yactual, ypredict):
        SST = np.sum((yactual - np.mean(yactual))**2) # total sum of squares
        SSR = np.sum((yactual - ypredict)**2)         # sum square residuals
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                rsq = 1. - SSR/SST
                cov = 100*np.sqrt((1./Rsq - 1.)/(len(yactual) - 2))
        except:
            rsq = 0.
            cov = 100.
        return {
            'rsq' : Rsq,
            'cov' : cov }
    # handle keyword arguments
    approximator = kwds.get('approximator',
                            approximate_elastic_regime_from_hough)
    # set the target optimization (minimization)
    target = lambda s : s['cov']
    if 'rsquared' in kwds:
        target = lambda s : 1. - s['rsq']
    # acceptable residual strain
    residual = kwds.get('residual', 0.00005)
    # #####  ##### #
    strain = mechprop.strain
    stress = mechprop.stress
    approx = approximator(mechprop)
    # use residual strain ($\epsilon - E \sigma$) to refine the
    # elastic properties.
    epsilon = approx['elastic strain']
    sigma   = approx['elastic stress']
    m       = approx['elastic modulus']
    b       = -m*approx['elastic onset']
    mask    = np.ones_like(epsilon)
    best = {
        'param' : [b, m],
        'cov'   : 1e6,
        'rsq'   : 1,
        'res'   : None,
        'mask'  : mask,
        'm'     : None
    }
    for numiter in range(maxiter):
        p0   = best['param']
        # perform a linear fit with the current set of points
        func = lambda param : epsilon[mask] - (sigma[mask] - param[0])/param[1]
        p, covp, infodict, mesg, ier = leastsq(func, p0, full_output=True)
        # the current modulus and intercept
        intercept, modulus = p
        # use residual strain to figure out the appropriate strain
        res = residual_strain(strain, stress, modulus)
        mask = (res < error)
        for i in reversed(range(len(res))):
            if mask[i]:
                mask[:i] = True
                break
        # statistics of the fit
        statistics = stats(sigma[mask], modulus*epsilon[mask] + intercept)
        # update the best
        if target(statistics) < target(best):
            best['param']   = p
            best['covx']    = covx
            best['cov']     = statistics['cov']
            best['rsq']     = statistics['rsq']
            best['res']     = res
            best['mask']    = mask
    # update the modulus of the mechanical properties
    intercept, modulus = best['param']
    mechprop.elastic_modulus = modulus
    mechprop.elastic_onset   = -intercept/modulus
