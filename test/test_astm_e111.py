# -*- coding: utf-8 -*-
from __future__ import division

import os, sys
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..'))

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
    HoughSpace,
    Normalized,
    approximate_elastic_regime_from_hough)
from citrine_converters.astm_e111.mechanical import linear_merge

STRAIN="{}/data/aramis-ey_strain-with-time.json".format(HERE)
STRESS="{}/data/mts-with-stress.json".format(HERE)

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


@pytest.fixture
def generate_output():
    return False


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


@pytest.fixture
def mechanical_properties(generate_output,
                          strain_dataframe,
                          stress_dataframe):
    strain = strain_dataframe
    stress = stress_dataframe
    return MechanicalProperties(strain, stress)


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
                                strain_dataframe,
                                stress_dataframe):
    strain = strain_dataframe
    stress = stress_dataframe
    mechprop = MechanicalProperties(strain, stress)
    assert np.isclose(mechprop.time.min(), 0.211914063), \
        "MechanicalProperties constructor intersection failed on lower bound."
    assert np.isclose(mechprop.time.max(), 245.00000), \
        "MechanicalProperties constructor intersection failed on upper bound."
    if generate_output:
        print "time(min, max) = ({:.3f}, {:.3f})".format(mechprop.time.min(),
                                                         mechprop.time.max())
        print "strain time(min, max) = ({:.3f}, {:.3f})".format(
            strain['time'].values.min(), strain['time'].values.max())
        print "stress time(min, max) = ({:.3f}, {:.3f})".format(
            stress['time'].values.min(), stress['time'].values.max())
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111)
        ax.plot(mechprop.strain, mechprop.stress, 'r-')
        ax.set_xlabel(r'$\epsilon$ (mm/mm)')
        ax.set_ylabel(r'$\sigma$ (MPa)')
        plt.draw()
        plt.savefig('{}/data/mechanical-constructor.png'.format(HERE),
            dpi=300, bbox_inches='tight')


def test_normalize(generate_output, mechanical_properties):
    mechprop = mechanical_properties
    strain = np.copy(mechprop.strain)
    nstrain = Normalized(strain)
    assert np.isclose(nstrain.min(), 0.0) and np.isclose(nstrain.max(), 1.0), \
        'Normalized not normalized [0, 1].'
    assert np.allclose(nstrain.unscaled, mechprop.strain), \
        'Unscaled normalized strain does not match original strain.'
    if generate_output:
        print "strain(min, max) = ({:.6f}, {:.6f})".format(
            mechprop.strain.min(), mechprop.strain.max())


def test_default_hough_constructor(generate_output, mechanical_properties):
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
                                               mechanical_properties):
    mechprop = mechanical_properties
    strain = mechprop.strain
    stress = mechprop.stress
    elastic = approximate_elastic_regime_from_hough(mechprop)
    assert np.isclose(elastic['elastic modulus'], 87448.9393194), \
        'Approximate elastic modulus does not match: ' \
        '{:.3f} (should be {:.3f})'.format(
            elastic['elastic modulus'],
            87448.9393194)
    assert np.isclose(elastic['elastic onset'], 7.240136402072304e-05), \
        'Approximate elastic onset does not match: '\
        '{:.g} (should be {:.3g})'.format(
            elastic['elastic onset'],
            7.240136402072304e-5)
    if generate_output:
        # grab data from elastic
        for k,v in elastic.iteritems():
            print "{}: ({}, {})".format(
                k, np.asarray(v).min(), np.asarray(v).max())
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
