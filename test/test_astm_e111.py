# -*- coding: utf-8 -*-
from __future__ import division

import os, sys
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..'))

import json
import pytest
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from bisect import bisect_left as bisect
from pypif import pif
from citrine_converters.astm_e111 import (
    MechanicalProperties,
    approximate_elastic_regime_from_hough,
    set_elastic)
from citrine_converters.astm_e111 import converter as astm_converter
from citrine_converters.tools import (
    HoughSpace,
    Normalized,
    linear_merge,
    covariance,
    r_squared)

STRAIN="{}/data/aramis-ey_strain-with-time.json".format(HERE)
STRESS="{}/data/mark10-with-stress.json".format(HERE)
EXPECTED="{}/data/expected-output.json".format(HERE)

def pif_to_dataframe(pifobj):
    """
    Converts the properties objects from a pif into a pandas.DataFrame object.

    Input
    =====
    :pifobj, pif: pif object

    Output
    ======
    pandas DataFrame constructed from pif.Properties objects in `pifobj`.
    """
    data = dict([(p.name, p.scalars) for p in pifobj.properties])
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def generate_output():
    return True


@pytest.fixture
def strain_dataframe():
    with open(STRAIN) as ifs:
        pifobj = pif.load(ifs)
    return pif_to_dataframe(pifobj)


@pytest.fixture
def stress_dataframe():
    with open(STRESS) as ifs:
        pifobj = pif.load(ifs)
    return pif_to_dataframe(pifobj)


@pytest.fixture(scope="module")
def expected_output(generate_output):
    if generate_output:
        write_output = True
        exdata = dict()
    else:
        write_output = False
        with open(EXPECTED, 'r') as ifs:
            exdata = json.load(ifs)
    yield exdata
    try:
        if write_output:
            with open(EXPECTED, 'w') as ofs:
                json.dump(exdata, ofs)
    except:
        print("Expected Output")
        print("---------------")
        for k,v in exdata.items():
            print("  {}: {}".format(k, v))
        raise


@pytest.fixture
def mechanical_properties(generate_output,
                          strain_dataframe,
                          stress_dataframe):
    strain = strain_dataframe
    stress = stress_dataframe
    return MechanicalProperties(strain, stress)


def test_r_squared():
    predicted = np.arange(15)
    actual = np.array([
        0.12772934,   0.87629623,   1.57547475,   2.65007133,   4.24801054,
        5.18072437,    6.2817513,   7.37424948,   7.97771838,   8.73240150,
        9.65820983,  10.67520638,  12.06516463,  12.95414000,  13.70856501])
    rsq = r_squared(predicted, predicted)
    assert np.isclose(1, rsq), \
        "Linear R^2 should be 1 ({})".format(rsq)
    rsq = r_squared(predicted, actual)
    assert np.isclose(0.996308546683, rsq), \
        "Random perturbation should be 0.996308546683 ({})".format(rsq)


def test_covariance():
    actual = np.arange(15)
    predicted = np.array(
        [ 0.09782940,   1.05192404,   2.02236946,   2.90373055,   3.92367251,
          4.92883409,   5.97735838,   6.92685796,   7.91508637,   8.95998904,
         10.01092281,  10.99958067,  12.08381427,  12.92445983,  14.03825360 ])
    cov = covariance(actual, actual)
    assert np.isclose(0, cov), \
        "Perfect covariance should be 0 ({}).".format(cov)
    cov = covariance(actual, predicted)
    assert np.isclose(0.4124658994, cov), \
        "Perfect covariance should be 0.4124658994 ({}).".format(cov)


def test_source_files(strain_dataframe, stress_dataframe):
    strain = strain_dataframe
    assert 'time' in strain.keys(), '"time" not found in strain data'
    assert 'strain' in strain.keys(), '"strain" not found in strain data'
    stress = stress_dataframe
    assert 'time' in stress.keys(), '"time" not found in stress data'
    assert 'stress' in stress.keys(), '"stress" not found in stress data'


def test_linear_merge():
    x1 = np.linspace(0.01, 1.234, num=30)
    x2 = np.linspace(0.02, 1.334, num=87)
    y1 = np.sin(x1)
    y2 = np.cos(x2)
    xm, y1m, y2m = linear_merge(x1, y1, x2, y2)
    assert xm.min() == x2.min(), "Intersection of x1/x2 failed on lower bound."
    assert xm.max() == x1.max(), "Intersection of x1/x2 failed on upper bound."
    assert xm.shape == (109,), "Intersection failed to produce 109 entries."
    assert xm.shape == y1m.shape and xm.shape == y2m.shape, \
        "Output shapes for linear merge do not match."


def test_mechanical_constructor(generate_output,
                                expected_output,
                                strain_dataframe,
                                stress_dataframe):
    strain = strain_dataframe
    stress = stress_dataframe
    mechprop = MechanicalProperties(strain, stress)
    if not generate_output:
        mechprop_time_min = expected_output['mechprop_time_min']
        mechprop_time_max = expected_output['mechprop_time_max']
        assert np.isclose(mechprop.time.min(), mechprop_time_min), \
            "MechanicalProperties constructor intersection failed on lower bound."
        assert np.isclose(mechprop.time.max(), mechprop_time_max), \
            "MechanicalProperties constructor intersection failed on upper bound."
    if generate_output:
        # record output
        expected_output['mechprop_time_min'] = mechprop.time.min()
        expected_output['mechprop_time_max'] = mechprop.time.max()
        expected_output['strain_time_min'] = strain['time'].values.min()
        expected_output['strain_time_max'] = strain['time'].values.max()
        expected_output['stress_time_min'] = stress['time'].values.min()
        expected_output['stress_time_max'] = stress['time'].values.max()
        # report output to stdout
        print("time(min, max) = ({:.3f}, {:.3f})".format(mechprop.time.min(),
                                                         mechprop.time.max()))
        print("strain time(min, max) = ({:.3f}, {:.3f})".format(
            strain['time'].values.min(), strain['time'].values.max()))
        print("stress time(min, max) = ({:.3f}, {:.3f})".format(
            stress['time'].values.min(), stress['time'].values.max()))
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111)
        ax.plot(mechprop.strain, mechprop.stress, 'r-')
        ax.set_xlabel(r'$\epsilon$ (mm/mm)')
        ax.set_ylabel(r'$\sigma$ (MPa)')
        plt.draw()
        plt.savefig('{}/data/mechanical-constructor.png'.format(HERE),
            dpi=300, bbox_inches='tight')


def test_normalize(generate_output, expected_output, mechanical_properties):
    mechprop = mechanical_properties
    strain = np.copy(mechprop.strain)
    nstrain = Normalized(strain)
    if not generate_output:
        nstrain_min = expected_output['nstrain_min']
        nstrain_max = expected_output['nstrain_max']
        assert np.isclose(nstrain.min(), nstrain_min) and \
               np.isclose(nstrain.max(), nstrain_max), \
            'Normalized not normalized [0, 1].'
        assert np.allclose(nstrain.unscaled, mechprop.strain), \
            'Unscaled normalized strain does not match original strain.'
    else:
        # record
        expected_output['nstrain_min'] = float(mechprop.strain.min())
        expected_output['nstrain_max'] = float(mechprop.strain.max())
        # report
        print("strain(min, max) = ({:.6f}, {:.6f})".format(
            mechprop.strain.min(), mechprop.strain.max()))


def test_default_hough_constructor(generate_output,
                                   mechanical_properties):
    mechprop = mechanical_properties
    h = HoughSpace(Normalized(np.copy(mechprop.strain)),
                   Normalized(np.copy(mechprop.stress)))
    # check hough
    assert np.allclose(mechprop.strain, h.x.unscaled), \
           'Strains do not match.'
    assert np.allclose(mechprop.stress, h.y.unscaled), \
           'Stresses do not match.'
    assert h.nq == 1801, \
        'Hough theta divided into {} divisions, should be 1801'.format(h.nq)
    assert h.nr == 1801, \
        'Hough radius divided into {} divisions, should be 1801'.format(h.nr)
    if generate_output:
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(16,16))
        ax = fig.add_subplot(111)
        im = ax.imshow(np.log(h + 1), cmap='jet')
        ax.grid()
        ax.set_xlabel(r'$r$')
        ax.set_ylabel(r'$\theta$')
        cb = plt.colorbar(im)
        cb.set_label('log(N)')
        plt.draw()
        plt.savefig('{}/data/default-hough.png'.format(HERE),
                    dpi=300, bbox_inches='tight')


def test_custom_hough_constructor(generate_output, mechanical_properties):
    mechprop = mechanical_properties
    h = HoughSpace(Normalized(np.copy(mechprop.strain)),
                   Normalized(np.copy(mechprop.stress)),
                   nq=721, nr=1001)
    # check hough
    assert h.nq == 721, \
        'Hough theta divided into {} divisions, should be 721'.format(h.nq)
    assert h.nr == 1001, \
        'Hough radius divided into {} divisions, should be 1001'.format(h.nr)
    assert np.allclose(mechprop.strain, h.x.unscaled), \
           'Strains do not match.'
    assert np.allclose(mechprop.stress, h.y.unscaled), \
           'Stresses do not match.'
    if generate_output:
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(16,16))
        ax = fig.add_subplot(111)
        im = ax.imshow(np.log(h + 1), cmap='jet')
        ax.grid()
        ax.set_xlabel(r'$r$')
        ax.set_ylabel(r'$\theta$')
        cb = plt.colorbar(im)
        cb.set_label('log(N)')
        plt.draw()
        plt.savefig('{}/data/custom-hough.png'.format(HERE),
                    dpi=300, bbox_inches='tight')


def test_approximate_elastic_regime_from_hough(generate_output,
                                               expected_output,
                                               mechanical_properties):
    mechprop = mechanical_properties
    strain = mechprop.strain
    stress = mechprop.stress
    elastic = approximate_elastic_regime_from_hough(mechprop)
    if not generate_output:
        elastic_modulus = expected_output['elastic modulus']
        elastic_onset = expected_output['elastic onset']
        assert np.isclose(elastic['elastic modulus'], elastic_modulus,
                          rtol=5.e-2), \
            'Approximate elastic modulus does not match: ' \
            '{:.3f} (should be {:.3f})'.format(
                elastic['elastic modulus'],
                elastic_modulus)
        assert np.isclose(elastic['elastic onset'], elastic_onset,
                          rtol=5.e-2, atol=0.0001), \
            'Approximate elastic onset does not match: ' \
            '{:} (should be {:.4g})'.format(
                elastic['elastic onset'],
                elastic_onset)
    else:
        # record
        expected_output['elastic modulus'] = float(elastic['elastic modulus'])
        expected_output['elastic onset'] = float(elastic['elastic onset'])
        # report
        for k,v in elastic.iteritems():
            print("{}: ({}, {})".format(
                k, np.asarray(v).min(), np.asarray(v).max()))
        epsilon = elastic['elastic strain']
        sigma = elastic['elastic stress']
        modulus = elastic['elastic modulus']
        onset = elastic['elastic onset']
        m = elastic['elastic modulus']
        b = -m*elastic['elastic onset']
        hough = elastic['hough']
        resampled = elastic['resampled']
        # plot annotated stress-strain curve
        x = [onset, (stress.max() - b)/m]
        y = [m*val + b for val in x]
        x2 = [onset + 0.002, (stress.max() - b)/m + 0.002]
        y2 = [m*(val-0.002) + b for val in x2]
        #
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111)
        ax.plot(strain, stress, 'k.')
        ax.plot(epsilon, sigma, 'bo')
        ax.plot(x, y, 'r-')
        ax.plot(x2, y2, 'g-')
        ax.set_xlabel(r'$\epsilon$ (mm/mm)')
        ax.set_ylabel(r'$\sigma$ (MPa)')
        plt.draw()
        plt.savefig('{}/data/approx-elastic-regime.png'.format(HERE),
                    dpi=300, bbox_inches='tight')
        #
        fig = plt.figure(figsize=(16,16))
        ax = fig.add_subplot(111)
        im = ax.imshow(np.log(resampled + 1), cmap='jet')
        cb = plt.colorbar(im)
        cb.set_label('log(N)')
        plt.draw()
        plt.savefig('{}/data/peaks-in-hough.png'.format(HERE),
                    dpi=300, bbox_inches='tight')


def test_set_elastic(generate_output,
                     expected_output,
                     mechanical_properties):
    mechprop = mechanical_properties
    strain = mechprop.strain
    stress = mechprop.stress
    best = set_elastic(mechprop)
    SE_slope = best['SE modulus']
    epstol = 5.e-2
    sigtol = 1.e-2
    if not generate_output:
        elastic_modulus = expected_output['set_elastic-elastic modulus']
        elastic_onset = expected_output['set_elastic-elastic onset']
        yield_stress = expected_output['set_elastic-yield stress']
        yield_strain = expected_output['set_elastic-yield strain']
        ultimate_stress = expected_output['set_elastic-ultimate stress']
        necking_onset = expected_output['set_elastic-necking onset']
        fracture_stress = expected_output['set_elastic-fracture stress']
        total_elongation = expected_output['set_elastic-total elongation']
        ductility = expected_output['set_elastic-ductility']
        toughness = expected_output['set_elastic-toughness']
        assert np.isclose(mechprop.elastic_modulus, elastic_modulus,
                          atol=SE_slope), \
            'Elastic modulus ({:.0f}) does not ' \
            'match ({:.0f})'.format(mechprop.elastic_modulus,
                                    elastic_modulus)
        assert np.isclose(mechprop.elastic_onset, elastic_onset,
                          rtol=epstol), \
            'Elastic onset ({:.4g}) does not ' \
            'match {:.4g}'.format(mechprop.elastic_onset,
                                  elastic_onset)
        assert np.isclose(mechprop.yield_stress, yield_stress,
                          atol=0.02*SE_slope), \
            'Yield stress ({:.3f}) does not ' \
            'match {:.3f}'.format(mechprop.yield_stress,
                                  yield_stress)
        assert np.isclose(mechprop.yield_strain, yield_strain,
                          rtol=epstol), \
            'Yield strain ({:.4g}) does not ' \
            'match {:.4g}'.format(mechprop.yield_strain,
                                  yield_strain)
        assert np.isclose(mechprop.ultimate_stress, ultimate_stress,
                          rtol=sigtol), \
            'Ultimate stress ({:.3f}) does not ' \
            'match {:.3f}'.format(mechprop.ultimate_stress,
                                  ultimate_stress)
        assert np.isclose(mechprop.necking_onset, necking_onset,
                          rtol=epstol), \
            'Necking onset ({:.4g}) does not ' \
            'match {:.4g}'.format(mechprop.necking_onset,
                                  necking_onset)
        assert np.isclose(mechprop.fracture_stress, fracture_stress,
                          rtol=sigtol), \
            'Fracture stress ({:.3f}) does not ' \
            'match {:.3f}'.fomat(mechprop.fracture_stress,
                                 fracture_stress)
        assert np.isclose(mechprop.total_elongation, total_elongation,
                          rtol=epstol), \
            'Total elongation ({:.4g}) does not ' \
            'match {:.4g}'.format(mechprop.total_elongation,
                                  total_elongation)
        assert np.isclose(mechprop.ductility, ductility,
                          rtol=epstol), \
            'Ductility ({:.4g}%) does not ' \
            'match {:.4g}%'.format(mechprop.ductility*100,
                                   ductility*100)
        assert np.isclose(mechprop.toughness, toughness,
                          rtol=sigtol), \
            'Toughness ({:.3f}) does not ' \
            'match {:.3f}'.format(mechprop.toughness,
                                  toughness)
    else:
        # record
        expected_output['set_elastic-fitting parameters'] = best['param']
        expected_output['set_elastic-STDEV modulus'] = best['SE modulus']
        expected_output['set_elastic-covariance'] = best['cov']
        expected_output['set_elastic-rsquared'] = best['rsq']
        expected_output['set_elastic-elastic modulus'] = mechprop.elastic_modulus
        expected_output['set_elastic-elastic onset'] = mechprop.elastic_onset
        expected_output['set_elastic-yield stress'] = mechprop.yield_stress
        expected_output['set_elastic-yield strain'] = mechprop.yield_strain
        expected_output['set_elastic-ultimate stress'] = mechprop.ultimate_stress
        expected_output['set_elastic-necking onset'] = mechprop.fracture_stress
        expected_output['set_elastic-total elongation'] = mechprop.total_elongation
        expected_output['set_elastic-ductility'] = mechprop.ductility
        expected_output['set_elastic-toughness'] = mechprop.toughness
        # report
        print('')
        print("best['param']: {}".format(best['param']))
        print("best['SE modulus']: {}".format(best['SE modulus']))
        print("best['cov']: {}".format(best['cov']))
        print("best['rsq']: {}".format(best['rsq']))
        print("modulus: {}".format(mechprop.elastic_modulus))
        print("onset: {}".format(mechprop.elastic_onset))
        print("yield stress: {}".format(mechprop.yield_stress))
        print("yield strain: {}".format(mechprop.yield_strain))
        print("plastic onset: {}".format(mechprop.plastic_onset))
        print("ultimate stress: {}".format(mechprop.ultimate_stress))
        print("necking onset: {}".format(mechprop.necking_onset))
        print("fracture stress: {}".format(mechprop.fracture_stress))
        print("total elongation: {}".format(mechprop.total_elongation))
        print("ductility: {}".format(mechprop.ductility))
        print("toughness: {}".format(mechprop.toughness))
        #
        plt.style.use('ggplot')
        #
        xoffset = 0.001
        yoffset = mechprop.elastic_modulus*xoffset/2
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111)
        ax.plot(strain, stress, 'k-', label=r'$\sigma(\epsilon)$')
        ax.plot(best['elastic strain'], best['elastic stress'], 'bx')
        # toughness
        mask = strain > mechprop.elastic_onset
        ax.fill_between(strain[mask], 0, stress[mask], facecolor='y', alpha=0.5)
        _ = ax.text((strain.max() + strain.min())/2, (stress.max() + stress.min())/2,
            'toughness = {:.3f} MPa'.format(mechprop.toughness))
        # Modulus curves:
        #+ set strain range (no offset)
        x = np.array([0, mechprop.ultimate_stress/mechprop.elastic_modulus])
        #+ calculate corresponding stress
        y = mechprop.elastic_modulus*x
        #+ perform strain to accound for elastic onset
        x += mechprop.elastic_onset
        ax.plot(x, y, 'r-', label='elastic')
        # 0.2% offset for plasticity
        x += 0.002
        ax.plot(x, y, 'g-', label='plastic')
        _ = ax.text(x[1] + xoffset, y[1],
            r'E = {:.3f} $\pm$ {:.3f} GPa'.format(
                mechprop.elastic_modulus/1000.,
                best['SE modulus']/1000),
            ha='left', va='center')
        # yield
        x = [mechprop.yield_strain]
        y = [mechprop.yield_stress]
        ax.scatter(x, y, marker='o', color='g', s=30, label='yield')
        _ = ax.text(x[0] + xoffset, y[0] - yoffset,
            r'$(\epsilon_y, \sigma_y)$ = ({:.0f} $\mu \epsilon$, {:.0f} MPa)'.format(1000*x[0], y[0]),
            ha='left', va='top')
        # ultimate
        x = [mechprop.necking_onset]
        y = [mechprop.ultimate_stress]
        ax.scatter(x, y, marker='^', color='r', s=30, label='ultimate')
        _ = ax.text(x[0], y[0] + yoffset,
            r'$(\epsilon_u, \sigma_u)$ = ({:.0f} $\mu \epsilon$, {:.0f} MPa)'.format(1000*x[0], y[0]),
            ha='center', va='bottom')
        # fracture
        x = [mechprop.total_elongation + mechprop.elastic_onset]
        y = [mechprop.fracture_stress]
        ax.scatter(x, y, marker='v', color='b', s=30, label='fracture')
        _ = ax.text(x[0], y[0]-yoffset,
            r'$(\epsilon_f, \sigma_f)$ = ({:.0f} $\mu \epsilon$, {:.0f} MPa)'.format(1000*x[0], y[0]),
            ha='right', va='top')
        # ductility
        x1 = mechprop.elastic_onset
        x2 = mechprop.ductility + mechprop.elastic_onset
        dx = x2 - x1
        y = yoffset
        dy = 0
        # ax.arrow(x1, y, dx, dy, fc='k', ec='k',
        #          head_width=10, head_length=0.005,
        #          overhang=0.1, length_includes_head=True)
        ax.annotate('',
            (x1, y), (x2, y),
            arrowprops=dict(
                arrowstyle='<->', lw=2,
                fc='k', ec='k'))
        ymin, ymax = ax.get_ylim()
        ax.axvline(x2,
                   ymin=(0 - ymin)/(ymax - ymin),
                   ymax=(2*y - ymin)/(ymax - ymin),
                   color='k')
        _ = ax.text((x1+x2)/2, y + yoffset,
            r'ductility = {:.3f}%'.format(mechprop.ductility*100),
            ha='center', va='bottom')
        #
        ax.set_xlabel(r'$\epsilon$ (mm/mm)')
        ax.set_ylabel(r'$\sigma$ (MPa)')
        plt.draw()
        plt.savefig('{}/data/mechanical_properties.png'.format(HERE),
                    dpi=300, bbox_inches='tight')


def test_converter(generate_output):
    astm_pif = astm_converter([STRAIN, STRESS])
    if generate_output:
        with open('{}/data/astm-mark10-aramis.json'.format(HERE), 'w') as ofs:
            pif.dump(astm_pif, ofs)
