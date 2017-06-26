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
from pypif import pif
from citrine_converters.visual import plot_stress_strain_from_pif as ssplot


@pytest.fixture
def save_output():
    return True


@pytest.fixture
def sspif():
    # stress-strain PIF
    with open('{}/data/astm-mark10-aramis.json'.format(HERE)) as ifs:
        rval = pif.load(ifs)
    return rval


def test_basic_stress_strain(save_output, sspif):
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 24
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    ssplot(sspif)
    if save_output:
        ofile = '{}/data/basic_stress_strain.png'.format(HERE)
        plt.draw()
        plt.savefig(ofile, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def test_all_stress_strain(save_output, sspif):
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 24
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    ssplot(sspif, all=True)
    if save_output:
        ofile = '{}/data/all_stress_strain.png'.format(HERE)
        plt.draw()
        plt.savefig(ofile, dpi=300, bbox_inches='tight')
    else:
        plt.show()
