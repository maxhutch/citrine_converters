# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from matplotlib import pyplot as plt


def plot_stress_strain_from_pif(obj,
        lc='k-',
        labeloffset=0.02,
        **kwds):
    """
    Plots a stress strain curve, optionally with:

      - fitting
      - elastic
      - yield
      - ultimate
      - fracture
      - ductility
      - toughness

    which are selected through keywords (described below), or
    by specifying the keyword `all` to plot everything.

    Examples:

        `plot_stress_strain_from_pif_object(mypif)`
        Plots a simple stress-strain curve with a black line.

        `plot_stress_strain_from_pif_object(mypif, yield='g--')`
        Plots a stress-strain curve with a black line and the
        0.2% offset yield line as a dashed green line.

    Required
    ========
    :param, obj: (pif.System object) stress strain data
        is stored in this object.

    Optional
    ========
    :param, lc: (matplotlib.LineStyle) matplotlib.plot
        line format indicator for the stress-strain curve.
        Default: 'k-' (black solid line)
    :param, labeloffset: (float) Fraction of the total plot
        offset to use in including text labels.
    :param, all: (bool) plot all annotations with their
        default settings.
    :param, fitting: (matplotlib.LineStyle) include points used
        in fitting the elastic region. Default: 'bx'
    :param, elastic: (matplotlib.LineStyle) include a
        line that shows the best fit elastic line. Default: 'r-'
    :param, yield: (matplotlib.LineStyle) include a
        line that shows the 0.2% offset yield. This also
        adds a labeled marker at the yield locus.
    :param, ultimate: (matplotlib.LineStyle) include
        a point at the location of the ultimate stress.
    :param, fracture: (matplotlib.LineStyle) include
        a labeled point at the location of the fracture stress.
    :param, ductility: (bool) include a arrow-ended
        line segment demarcating the ductility.
    :param, toughness: (matplotlib.color) highlight the region
        under the curve describing toughness. The string

    Output
    ======
    :return: matplotlib.Figure
    """
    def get_property(pobj, name):
        try:
            return [p for p in pobj.properties
                    if p.name == name][0]
        except IndexError:
            return None

    # handle defaults
    if_not_set = lambda key, default: \
        default if key not in kwds else kwds[key]
    if 'all' in kwds:
        kwds['elastic']   = if_not_set('elastic', 'r-')
        kwds['fitting']   = if_not_set('fitting', 'bx')
        kwds['yield']     = if_not_set('yield', 'g-')
        kwds['ultimate']  = if_not_set('ultimate', 'b')
        kwds['fracture']  = if_not_set('fracture', 'r')
        kwds['ductility'] = if_not_set('ductility', True)
        kwds['toughness'] = if_not_set('toughness', 'y')

    # get data from the pif object
    elastic_modulus = get_property(obj, 'elastic modulus').scalars
    m, dm = elastic_modulus.value, elastic_modulus.uncertainty
    elastic_onset = get_property(obj, 'elastic onset').scalars
    yield_strength = get_property(obj, 'yield strength').scalars
    yield_strain = get_property(obj, 'yield strain').scalars
    ultimate_strength = get_property(obj, 'ultimate strength').scalars
    necking_onset = get_property(obj, 'necking onset').scalars
    fracture_strength = get_property(obj, 'fracture strength').scalars
    total_elongation = get_property(obj, 'total elongation').scalars
    toughness = get_property(obj, 'toughness').scalars
    ductility = get_property(obj, 'ductility').scalars
    covariance = get_property(obj, 'covariance').scalars
    rsquared = get_property(obj, 'coefficient of variation').scalars
    mask = np.array(get_property(obj, 'fitting mask').scalars)
    # set stress-strain
    strain = np.array(get_property(obj, 'strain').scalars) - elastic_onset
    stress = np.array(get_property(obj, 'stress').scalars)
    elastic_strain = np.array(get_property(obj, 'elastic strain').scalars) - elastic_onset
    elastic_stress = np.array(get_property(obj, 'elastic stress').scalars)

    # set appearance
    # plt.style.use(style)
    # plt.rcParams['font.size'] = fontsize

    # create the figure
    ax = plt.gca()
    # stress-strain
    ax.plot(strain, stress, lc, label=r'$\sigma(\epsilon)$')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xoffset = (xlim[1] - xlim[0])*labeloffset
    yoffset = (ylim[1] - ylim[0])*labeloffset
    # fitting
    if 'fitting' in kwds:
        ax.plot(elastic_strain[mask], elastic_stress[mask],
                kwds['fitting'], label='fittings')
        _ = ax.text(0.95, 0.25,
                    'Covariance: {cov:.3f}\n' \
                    '$R^2$: {r2:.3f}\n'.format(
                        cov=covariance, r2=rsquared),
                    ha='right', va='center',
                    transform=ax.transAxes)
    # toughness
    if 'toughness' in kwds:
        mask = (strain > 0)
        ax.fill_between(strain[mask], 0, stress[mask],
                        facecolor=kwds['toughness'],
                        alpha=0.5)
        _ = ax.text((strain.max() + strain.min())/2,
                    (stress.max() + stress.min())/2,
                    'toughness = {:.3f} MPa'.format(toughness))
    # elastic
    if 'elastic' in kwds:
        x = np.array([0, ultimate_strength/m])
        y = m*x
        im = ax.plot(x, y, kwds['elastic'], label='elastic')
    # yield
    if 'yield' in kwds:
        x = np.array([0, ultimate_strength/m])
        y = m*x
        x += 0.002
        (im,) = ax.plot(x, y, kwds['yield'], label='yield')
        c = im.get_color()
        _ = ax.text(x[1] + xoffset, y[1],
                r'E = {:.3f} $\pm$ {:.3f} GPa'.format(
                    m/1000., dm/1000.),
                ha='left', va='center')
        x = [yield_strain]
        y = [yield_strength]
        im = ax.scatter(x, y, marker='o', color=c, s=30, label='yield')
        _ = ax.text(x[0] + xoffset, y[0] - yoffset,
                r'$(\epsilon_y, \sigma_y)$ = ' \
                 '({:.0f} $\mu \epsilon$, {:.0f} MPa)'.format(
                        1000*x[0], y[0]),
                ha='left', va='top')
    # ultimate
    if 'ultimate' in kwds:
        x = [necking_onset]
        y = [ultimate_strength]
        ax.scatter(x, y, marker='^', color=kwds['ultimate'],
                s=30, label='ultimate')
        _ = ax.text(x[0], y[0] + yoffset,
                r'$(\epsilon_u, \sigma_u)$ = ' \
                 '({:.0f} $\mu \epsilon$, {:.0f} MPa)'.format(
                        1000*x[0], y[0]),
                ha='center', va='bottom')
    # fracture
    if 'fracture' in kwds:
        x = [total_elongation]
        y = [fracture_strength]
        ax.scatter(x, y, marker='v', color=kwds['fracture'],
                s=30, label='fracture')
        _ = ax.text(x[0], y[0]-yoffset,
            r'$(\epsilon_f, \sigma_f)$ = ' \
             '({:.0f} $\mu \epsilon$, {:.0f} MPa)'.format(
                    1000*x[0], y[0]),
            ha='right', va='top')
    # ductility
    if 'ductility' in kwds:
        x1, x2 = 0, ductility
        dx = x2 - x1
        y = yoffset
        dy = 0
        ax.annotate('',
                (x1, y), (x2, y),
                arrowprops=dict(
                        arrowstyle='<->', lw=2,
                        fc='k', ec='k'))
        ax.axvline(x2,
                ymin=(0 - ylim[0])/(ylim[1] - ylim[0]),
                ymax=(2*y - ylim[0])/(ylim[1] - ylim[0]),
                color='k')
        _ = ax.text((x1 + x2)/2, y + yoffset,
                r'ductility = {:.3f}%'.format(ductility*100),
                ha='center', va='bottom')
    # set the plot labels
    units = get_property(obj, 'strain').units
    ax.set_xlabel(r'$\epsilon$ (${}$)'.format(units))
    units = get_property(obj, 'stress').units
    ax.set_ylabel(r'$\sigma$ (${}$)'.format(units))
    # done
    return ax
