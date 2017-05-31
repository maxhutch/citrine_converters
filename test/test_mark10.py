# -*- coding: utf-8 -*-

import os, sys
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..'))

import pytest
from pypif import pif
from citrine_converters.mark10 import converter


SOURCE="{}/data/mark10-output.csv".format(HERE)


@pytest.fixture
def generate_output():
    return False


@pytest.fixture
def mark10_no_stress():
    with open('{}/data/mark10-no-stress.json'.format(HERE)) as ifs:
        expected = ifs.read()
    return expected.strip()


@pytest.fixture
def mark10_with_stress():
    with open('{}/data/mark10-with-stress.json'.format(HERE)) as ifs:
        expected = ifs.read()
    return expected.strip()


def test_file_only(mark10_no_stress, generate_output):
    pifs = converter(SOURCE)
    if generate_output:
        with open('{}/data/mark10-no-stress.json'.format(HERE), 'w') as ofs:
            pif.dump(pifs, ofs, sort_keys=True)
        assert False
    pifs = pif.dumps(pifs, sort_keys=True).strip()
    assert pifs == mark10_no_stress


def test_file_list(mark10_no_stress, generate_output):
    pifs = converter([SOURCE])
    if generate_output:
        with open('{}/data/mark10-no-stress.json'.format(HERE), 'w') as ofs:
            pif.dump(pifs, ofs, sort_keys=True)
        assert False
    pifs = pif.dumps(pifs, sort_keys=True).strip()
    assert pifs == mark10_no_stress


def test_stress(mark10_with_stress, generate_output):
    area=12.9
    units='mm^2'
    pifs = converter(SOURCE, area=area)
    if generate_output:
        with open('{}/data/mark10-with-stress.json'.format(HERE), 'w') as ofs:
            pif.dump(pifs, ofs, sort_keys=True)
        assert False
    pifs = pif.dumps(pifs, sort_keys=True).strip()
    assert pifs == mark10_with_stress
