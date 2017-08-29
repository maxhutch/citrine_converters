# -*- coding: utf-8 -*-

from pypif import pif
from StringIO import StringIO
import numpy as np
import pandas as pd
from ..tools import replace_if_present_else_append


def __can_convert(line, sep=','):
    try:
        _ = [float(word) for word in line.strip().split(sep)]
        return True
    except ValueError:
        return False


def __andrew_peterson_ca2015(filename, **keywds):
    """
    Summary
    =======

    Reads results stored by the MATLAB script written by
    Andrew Peterson at Colorado School of Mines in ca. 2015.
    This file does not contain any header information, but
    stores data in this order:

    | column |    label     | units |
    |--------|--------------|-------|
    |    1   |     time     |   s   |
    |    2   |    force     |   N   |
    |    3   | displacement |  mm   |
    |    4   |    stress    |  MPa  |

    The last row is optional.

    Because this is a legacy format, many of these have been
    modified manually to include header information. The
    format of the header information is unknown and user-
    specific. Therefore, this method strips all header
    information that may exist and assumes the original
    order.

    :param, filename: (str) Filename to read.
    :param, names: (list, optional) List of column names.
    :param, units: (dict, optional) Dictionary of column
        units. The keys should match `names` or the default
        column labels in the table above.
    :return: (labels, units, pandas.DataFrame)
    """
    # open the file for reading and extract only those
    # lines that can be converted into floats.
    with open(filename) as ifs:
        lines = [line for line in ifs.readlines()
                 if __can_convert(line)]
    # How many fields are present? Some files have
    # extraneous rows with information on creation date/time,
    # and other numerical data that, while convertable to
    # float, are not measurement data. Since these are
    # only inconsistently present, and a user-specific
    # addition, they should be ignored. The median number of
    # fields eliminates these spurious rows.
    nfields = np.median(
        [len(line.strip().split(',')) for line in lines]).astype(int)
    if nfields > 4:
        msg = 'Up to four fields are recognized for Mark 10 output. ' \
              'Found {} fields.'.format(nfields)
        raise IOError(msg)
    # if names were provided use them.
    keywds['names'] = keywds.get('names',
        ['time', 'force', 'displacement', 'stress'][:nfields])
    keywds['names'] = [name.lower() for name in keywds['names']]
    # check if units were provided
    units = {
        'time' : 's',
        'force' : 'N',
        'displacement' : 'mm',
        'stress' : 'MPa' }
    # modify units to work on the case-insensitive keys
    if 'units' in keywds:
        for k,v in iter(keywds['units'].items()):
            units[k.lower()] = v
        del keywds['units']
    # read in the data
    sio = StringIO(''.join(lines))
    data = pd.read_csv(sio, **keywds)
    names = tuple(data.columns.values)
    units = tuple(units.get(name, 'unknown') for name in names)
    # return the file information
    return (names, units, data)


def converter(files=[], **keywds):
    """
    Summary
    =======

    Converter to calculate stress data from Mark10 CSV output.
    The format is dependent on the script used to collect the data
    from the ASCII output of the Mark 10. Andrew Peterson wrote
    the MATLAB script that captures the data from the Mark10 at
    CSM. In order to maintain generality, this script keeps only
    the numerical data. The columns are assumed to be "time",
    "force", "displacement", and optionally, "stress". Column
    names and units may be provided expressly through the keywords
    argument.

    :param, files: (list) List of CSV-formatted files.
    :param, area: (float, optional) Cross sectional area of the
        sample.
    :param, units: (dict, optional)  Units (strings) for each name,
        either read from the keywds (case insensitive) or defaults.
    
    All other keywords are passed to pandas.read_csv. (Note:
    `pandas.read_csv` does not handle unknown keywords
    gracefully. An unknown keyword will throw an error.)

    :return: PIF object or list of PIF objects
    """
    # ensure *files* is a list. If a single file is passed, convert it
    # into a list.
    if isinstance(files, str):
        files = [files]
    # handle optional arguments
    # check if area was provided
    if 'area' in keywds:
        area = float(keywds['area'])
        area_units = keywds.get('units', {}).get('area', None)
        del keywds['area']
    else:
        area = None
    # process files
    results = []
    for fname in files:
        names, units, data = __andrew_peterson_ca2015(fname, **keywds)
        # list of properties extracted from the file
        for name,unit in zip(names, units):
            replace_if_present_else_append(results,
                pif.Property(
                    name=name,
                    scalars=list(data[name]),
                    units=unit,
                    files=pif.FileReference(relative_path=fname),
                    methods=pif.Method(name='uniaxial',
                        instruments=pif.Instrument(producer='Mark10')),
                    data_type='EXPERIMENTAL',
                    tag='Mark10'),
                cmp=lambda a,b: a.name.lower() == b.name.lower())
        # Calculate stress from force and cross-sectional area,
        # if provided.
        if ('force' in names) and (area is not None):
            unit_dict = dict(zip(names, units))
            force_units = unit_dict.get('force', 'unknown')
            displacement_units = unit_dict.get('displacement', 'unknown')
            area_units = area_units if area_units is not None else \
                '{}^2'.format(displacement_units)
            stress_units = unit_dict.get('stress',
                '{}/{}'.format(force_units, area_units))
            # add property to results
            replace_if_present_else_append(results,
                pif.Property(
                    name='area',
                    scalars=area,
                    units=area_units,
                    files=pif.FileReference(relative_path=fname),
                    methods=pif.Method(name='uniaxial',
                        instruments=pif.Instrument(producer='Mark10')),
                    data_type='EXPERIMENTAL',
                    tag='cross sectional area'),
                cmp=lambda A,B : A.name.lower() == B.name.lower())
            replace_if_present_else_append(results,
                pif.Property(
                    name='stress',
                    scalars=list(data['force']/area),
                    units=stress_units,
                    files=pif.FileReference(relative_path=fname),
                    methods=pif.Method(name='uniaxial',
                        instruments=pif.Instrument(producer='Mark10')),
                    data_type='EXPERIMENTAL',
                    tag='Mark10'),
                cmp=lambda A,B : A.name.lower() == B.name.lower())
    # Wrap in system object
    results = pif.System(
        names='Mark10',
        properties=results,
        tags=files)
    # job's done!
    return results
