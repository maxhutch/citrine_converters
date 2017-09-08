# -*- coding: utf-8 -*-

from .mechanical import MechanicalProperties, set_elastic
from pypif import pif
import re
import pandas as pd
import numpy as np


def converter(files=[], **keywds):
    """
    Summary
    =======

    Accepts PIF formatted stress and strain data and calculates the
    following mechanical information from these data:

        - elastic (Young's) modulus
        - elastic onset (strain)
        - yield strength
        - yield strain (plastic onset)
        - ultimate strength
        - necking onset (strain)
        - fracture strength
        - total elongation
        - ductility
        - toughness

    The input PIF files must contain "time" and "stress" fields (stress
    PIF) and "time" and "strain" fields (strain PIF).

    Input
    =====
    :param files, list: `[stress_filename, strain_filename]` where
        `stress_filename` and `strain_filename` are the filenames of the
        stress and strain data, respectively.

    Keywords
    ========
    :units, string: unit convention used in this stress-strain data.
        Should be one of: {MPa, kip}

    Output
    ======
    PIF object.
    """
    # Handle input parameters
    #+ ensure two files were provided
    try:
        left, right = files
    except ValueError:
        msg = 'Converter requires stress and strain filenames.'
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
        'time' in [prop.name for prop in right.properties]), \
        "Both strain and stress must have synchronized time data."
    if 'strain' in [prop.name for prop in left.properties]:
        subsys = {'strain' : left, 'stress' : right}
        strain, stress = left.properties, right.properties
    else:
        subsys = {'stress' : left, 'strain' : right}
        stress, strain = left.properties, right.properties
    try:
        # create pandas dataframe for strain data
        x = [prop.scalars for prop in strain if prop.name == 'time'][0]
        y = [prop.scalars for prop in strain if prop.name == 'strain'][0]
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
        msg = 'Strain and stress files must contain "strain" and ' \
              '"stress" fields, respectively.'
        raise IndexError(msg)

    # TODO: This may be a good use case for a more general set of
    # `transform` modules, e.g. `transform.reflect`,
    # `transform.scale`, etc. For only one, this is overkill, but
    # if more transforms are necessary to handle more edge cases,
    # implement replace this with
    # `..tools.transform.reflect(np.sign(np.mean(vec)))` and
    # implement any future transforms in a similar fashion.
    #
    # Strain can be recorded as negative for compression, but this
    # is non-standard. Reverse the direction of the strain if it
    # moves negatively
    # ensure the strain data progresses in the +x direction
    vec = epsilon['strain'].values
    steps = vec[1:] - vec[:-1]
    # -1 --> reflection, 1 --> identity
    reflect = np.sign(np.mean(steps))
    epsilon['strain'] *= reflect
    # similaly ensure the stress data progresses in the +y direction
    # i.e. stress is positive. Compression/tension distinguished by
    # the direction of loading.
    vec = sigma['stress'].values
    steps = vec[1:] - vec[:-1]
    # -1 --> reflection, 1 --> identity
    reflect = np.sign(np.mean(steps))
    sigma['stress'] *= reflect

    # units
    try:
        # units explicitly given supercede stress/strain input
        # does this make sense?
        units = keywds['units'].lower()
        if units == 'mpa':
            stress_units = 'MPa'
            strain_units = 'mm/mm'
        elif units in ('kip', 'kips'):
            stress_units = 'kip'
            strain_units = 'in/in'
        else:
            stress_units = 'unknown'
            strain_units = 'unitless'
    except KeyError:
        # if not specified, then get from stress/strain input
        try:
            stress_units = [p.units for p in stress if p.name == 'stress'][0]
            strain_units = [p.units for p in strain if p.name == 'strain'][0]
        except IndexError:
            stress_units = 'unknown'
            strain_units = 'unitless'

    # TODO: add epsilon/sigma mask to property results

    # Calculate the mechanical properties of the object
    mechprop = MechanicalProperties(epsilon, sigma)
    best = set_elastic(mechprop)
    SE_modulus = best['SE modulus']
    # Create the PIF file
    results = [
        pif.Property(name='strain',
            scalars=list(mechprop.strain),
            units=strain_units),
        pif.Property(name='stress',
            scalars=list(mechprop.stress),
            units=stress_units),
        pif.Property(name='elastic strain',
            scalars=list(best['elastic strain']),
            units=strain_units),
        pif.Property(name='elastic stress',
            scalars=list(best['elastic stress']),
            units=stress_units),
        pif.Property(name='fitting mask',
            scalars=list(best['mask'].astype(int)),
            units='unitless',
            data_type='FIT',
            tags='Mask of elastic stress/strain data used in the fitting'),
        pif.Property(name='covariance',
            scalars=best['cov'],
            units='unitless',
            data_type='FIT',
            tags='COV of the linear elastic fit'),
        pif.Property(name='coefficient of variation',
            scalars=best['rsq'],
            units='unitless',
            data_type='FIT',
            tag=r'$R^2$ of the linear elastic fit'),
        pif.Property(name='elastic modulus',
            scalars=pif.Scalar(value=mechprop.elastic_modulus,
                               uncertainty=SE_modulus),
            units=stress_units,
            data_type='FIT'),
        pif.Property(name='elastic onset',
            scalars=mechprop.elastic_onset,
            units=strain_units,
            data_type='FIT'),
        pif.Property(name='yield strength',
            scalars=mechprop.yield_stress,
            units=stress_units,
            data_type='FIT'),
        pif.Property(name='yield strain',
            scalars=mechprop.yield_strain,
            units=strain_units,
            data_type='FIT'),
        pif.Property(name='ultimate strength',
            scalars=mechprop.ultimate_stress,
            units=stress_units,
            data_type='FIT'),
        pif.Property(name='necking onset',
            scalars=mechprop.necking_onset,
            units=strain_units,
            data_type='FIT'),
        pif.Property(name='fracture strength',
            scalars=mechprop.fracture_stress,
            units=stress_units,
            data_type='FIT'),
        pif.Property(name='total elongation',
            scalars=mechprop.total_elongation,
            units=strain_units,
            data_type='FIT'),
        pif.Property(name='ductility',
            scalars=mechprop.ductility,
            units=strain_units,
            data_type='FIT'),
        pif.Property(name='toughness',
            scalars=mechprop.toughness,
            units=stress_units,
            data_type='FIT')
    ]
    # Wrap in system object
    hough = best['hough']
    resampled = best['resampled']
    results = pif.System(
        names='stress-strain curve',
        sub_systems=[subsys['strain'], subsys['stress']],
        preparation=pif.ProcessStep(
            name='approximator',
            details=[
                pif.Value(
                    name='hough',
                    vectors=[[col for col in row] for row in hough]
                ),
                pif.Value(
                    name='resampled hough',
                    vectors=[[col for col in row] for row in resampled]
                )]),
        properties=results,
        references=pif.Reference(
            url='https://www.astm.org/Standards/E111.htm'),
        tags=['ASTM E111'])
    # job's done!
    return results
