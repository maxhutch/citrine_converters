# -*- coding: utf-8 -*-

from .mechanical import MechanicalProperties
from pypif import pif
import re
import pandas as pd


def converter(files=[], **keywds):
    """
    Summary
    =======

    Accepts PIF formatted stress and strain data and calculates the
    following mechanical information from these data:

        - Elastic (Young's) modulus
        - Yield strength
        - Strain at yield
        - Ultimate strength

    The input PIF files must contain "time" and "stress" fields (stress
    PIF) and "time" and "epsilon y" fields (strain PIF).

    IN
    ==
    :param files, list: `[stress_filename, strain_filename]` where
        `stress_filename` and `strain_filename` are the filenames of the
        stress and strain data, respectively.
    :param keywds: None

    OUT
    ===
    PIF object or list of PIF objects.
    """
    # Handle input parameters
    #+ ensure two files were provided
    try:
        left, right = files
    except ValueError:
        msg = 'Converter requires a stress an a strain filename.'
        raise ValueError(msg)
    # Read files
    try:
        left = pif.load(open(left))
        right = pif.load(open(right))
    except JSONDecodeError:
        msg = 'Stress or strain data is not a properly formatted PIF file.'
        raise IOError(msg)
    #+ ensure strain file has "time" and "epsilon y"
    #+ ensure stress file has "time" and "stress"
    assert (
        'time' in [prop.name for prop in left.properties] and
        'time' in [prop.name for prop in right.properties]) \
        "Both strain and stress must have synchronized time data."
    if 'epsilon y' in [prop.name for prop in left.properties]:
        strain, stress = left.properties, right.properties
    else:
        stress, strain = left.properties, right.properties
    try:
        # create pandas dataframe for strain data
        x = [prop.scalars for prop in strain if prop.name == 'time'][0]
        y = [prop.scalars for prop in strain if prop.name == 'epsilon y'][0]
        epsilon = pd.DataFrame({
            'time' : x,
            'strain' : y})
        # create pandas dataframe for stress data
        x = [prop.scalars for prop in stress if prop.name == 'time'][0]
        y = [prop.scalars for prop in stress if prop.name == 'stress'][0]
        sigma = pd.DataFrame({
            'time' : x,
            'stress' : y})
    except IndexError:
        msg = 'Strain and stress files must contain "epsilon y" and ' \
              '"stress" fields, respectively.'
        raise IndexError(msg)

    # Calculate the mechanical properties of the object
    mechprop = MechanicalProperties(epsilon, sigma)

    # TODO Create an elastic modulus property

    # TODO Create a yield strength property

    # TODO Create a strain at yield property

    # TODO Create an ultimate strength property

    # job's done!
    return results
