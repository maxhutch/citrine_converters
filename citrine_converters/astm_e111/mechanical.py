from __future__ import division

import numpy as np
import warnings
from scipy.ndimage import gaussian_filter
from scipy.optimize import leastsq
from scipy.stats import linregress
from ..tools import (
    linear_merge,
    Normalized,
    resample,
    HoughSpace,
    r_squared,
    covariance)


ELASTIC_OFFSET=0.002


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
        #+ use strain values that are greater that 0.2% above
        #+ the elastic onset
        subset = (self.strain > onset)
        substrain = self.strain[subset]
        substress = self.stress[subset]
        #+ find the hypothetical elastic stress at each strain
        estress = modulus*(substrain - self.elastic_onset - ELASTIC_OFFSET)
        #+ when the hypothetical elastic stress is greater than
        #+ the measured stress, we have reached yield.
        mask   = (substress < estress)
        for i in reversed(range(len(mask)-1)):
            if not mask[i]:
                yield_stress = (substress[i] + substress[i+1])/2
                break
        return yield_stress

    @property
    def yield_strain(self):
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
        yield_stress = self.yield_stress
        yield_strain = yield_stress/modulus + ELASTIC_OFFSET
        return yield_strain

    @property
    def plastic_onset(self):
        return self.yield_strain

    @property
    def elastic_region(self):
        onset = self.elastic_onset
        if onset is None:
            msg = 'The elastic onset must be set before the ' \
                  'elastic region can be defined.'
            raise ValueError(msg)
        try:
            yield_strain = self.yield_strain
        except ValueError:
            msg = 'The elastic modulus must be set before the ' \
                  'elastic region can be defined.'
        strain = self.strain
        mask = (strain > onset) & (strain < yield_strain + onset)
        return self.strain[mask]

    @property
    def plastic_region(self):
        try:
            yield_strain = self.yield_strain
        except ValueError:
            msg = 'The elastic modulus must be set before the ' \
                  'elastic region can be defined.'
        strain = self.strain
        mask = (strain > yield_strain + onset)
        return self.strain[mask]

    @property
    def ultimate_stress(self):
        return self.stress.max()

    @property
    def necking_onset(self):
        i = np.where(self.stress == self.ultimate_stress)[0][0]
        return self.strain[i] - self.elastic_onset

    @property
    def fracture_stress(self):
        i = np.argmax(self.strain)
        return self.stress[i]

    @property
    def total_elongation(self):
        i = np.argmax(self.strain)
        return self.strain[i] - self.elastic_onset

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
        mask = strain > self.elastic_onset
        return np.dot((stress[mask][1:] + stress[mask][:-1])/2.,
                      (strain[mask][1:] - strain[mask][:-1]))
#end 'class MechanicalProperties(object):'


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
        y = (x-ELASTIC_OFFSET) \tan(\theta) \left( \frac{y_{max} - y_{min}}{x_{max} - x_{min}} \right) - x_{min} \tan(\theta)\left( \frac{y_{max} - y_{min}}{x_{max} - x_{min}} \right) - d (y_{max} - y_{min}) \sec(\theta) + y_{min}
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
    # find the plastic region (lies below `y = m*(x - ELASTIC_OFFSET) + b` line)
    plastic = (y < (m*(x - ELASTIC_OFFSET) + b))
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
    :error, float: strain gage measurement error. Default: 0.00005.

    Output
    ======
    Returns best performance metrics. Modulus and offset stored in
    `mechprop`.
    """
    # handle keyword arguments
    approximator = kwds.get('approximator',
                            approximate_elastic_regime_from_hough)
    # maximum number of iterations
    maxiter = kwds.get('maxiter', 20)
    # set the target optimization (minimization)
    target = lambda cov, rsq : cov
    if 'cov' in kwds:
        target = lambda cov, rsq : cov
    elif 'rsq' in kwds:
        target = lambda cov, rsq : 1. - rsq
    # error in the strain gage measurement
    error = kwds.get('error', 0.00005)
    # #####  ##### #
    strain = mechprop.strain
    stress = mechprop.stress
    approx = approximator(mechprop)
    # sigma = E epsilon + offset
    # epsilon = e_calc = (sigma - offset)/E
    # t.f.
    # residual strain = e_measured - e_calc
    predicted_strain = lambda s, E, b : (s - b)/E
    residual_strain = lambda e, s, E, b: e - predicted_strain(s, E, b)
    # use residual strain ($\epsilon - E \sigma$) to refine the
    # elastic properties.
    epsilon = approx['elastic strain']
    sigma   = approx['elastic stress']
    m       = approx['elastic modulus']
    b       = -m*approx['elastic onset']
    mask    = np.ones_like(epsilon, dtype=bool)
    res     = residual_strain(epsilon, sigma, m, b)
    best = {
        'param' : [b, m],
        'SE modulus'  : 1e6,
        'cov'   : covariance(predicted_strain(sigma, m, b), epsilon),
        'rsq'   : r_squared(predicted_strain(sigma, m, b), epsilon),
        'res'   : res,
        'elastic strain' : epsilon,
        'elastic stress' : sigma,
        'mask'  : mask
    }
    # # update the modulus of the mechanical properties
    # intercept, modulus = best['param']
    # mechprop.elastic_modulus = modulus
    # mechprop.elastic_onset   = -intercept/modulus
    # return best
    # perform a linear fit with the current set of points
    fitfunc = lambda p : \
        residual_strain(epsilon[mask], sigma[mask], p[1], p[0])
    for numiter in range(maxiter):
        # p0   = best['param']
        # p, covp, infodict, mesg, ier = leastsq(
        #     fitfunc, p0, full_output=True)
        # the current modulus and intercept
        # intercept, modulus = p
        modulus, intercept, corrcoeff, pvalue, SE_slope = \
            linregress(epsilon[mask], sigma[mask])
        # use residual strain to figure out the appropriate
        # points to use in calculating the stress/strain
        res = residual_strain(epsilon, sigma, modulus, intercept)
        mask = (np.abs(res) < error)
        for i in reversed(range(len(mask)-1)):
            if mask[i]:
                upper = i+1
                break
        mask[upper:] = False
        mask[:upper] = True
        # statistics of the fit
        try:
            subset = residual_strain(
                epsilon[mask], sigma[mask], modulus, intercept)
            calc = predicted_strain(sigma[mask], modulus, intercept)
            cov = covariance(calc, epsilon[mask])
            rsq = corrcoeff**2
            # rsq = r_squared(calc, epsilon[mask])
        except ValueError: # invalid number of observations
            continue
        # update the best
        if target(cov, rsq) < target(best['cov'], best['rsq']):
            best['param']   = [intercept, modulus]
            best['SE modulus'] = SE_slope
            best['cov']     = cov
            best['rsq']     = rsq
            best['res']     = res
            best['mask']    = mask
            best['elastic strain'] = epsilon[mask]
            best['elastic stress'] = sigma[mask]
    # update the modulus of the mechanical properties
    intercept, modulus = best['param']
    mechprop.elastic_modulus = modulus
    mechprop.elastic_onset   = -intercept/modulus
    # return the performance metrics
    return best
