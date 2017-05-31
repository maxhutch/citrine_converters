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


def converter(files=[], **keywds):
    """
    Summary
    =======

    Converter to calculate stress data from Mark10 CSV output.
    The format is dependent on the script used to collect the data
    from the ASCII output of the Mark 10.

    Input
    =====
    :files, list: List of CSV-formatted files.

    Options
    -------
    :area, float: Cross sectional area of the sample.
    :units, dict: Units (strings) for each name, either read
        from the keywds (case insensitive) or defaults.
    Other keywords are passed to pandas.read_csv.

    Output
    ======
    PIF object or list of PIF objects
    """
    # ensure *files* is a list. If a single file is passed, convert it
    # into a list.
    if isinstance(files, str):
        files = [files]
    for fname in files:
        with open(fname) as ifs:
            with open(fname) as ifs:
                lines = [line for line in ifs.readlines()
                         if __can_convert(line)]
            nfields = np.median(
                [len(line.strip().split(',')) for line in lines]).astype(int)
            assert nfields <= 4, \
                'Up to four fields are recognized for Mark 10 output. ' \
                'Found {} fields.'.format(nfields)
            # if names were provided use these.
            keywds['names'] = keywds.get(
                'names', ['time', 'force', 'displacement', 'stress'])
            keywds['names'] = [name.lower() for name in keywds['names']]
            keywds['names'] = keywds['names'][:nfields]
            # check if units were provided
            units = {
                'time' : 's',
                'force' : 'N',
                'displacement' : 'mm',
                'stress' : 'MPa' }
            if 'units' in keywds:
                for k,v in iter(keywds['units'].items()):
                    units[k.lower()] = v
                del keywds['units']
            units['area'] = units.get(
                'area', units['displacement'] + '^2')
            # check if area was provided
            if 'area' in keywds:
                area = float(keywds['area'])
                del keywds['area']
            else:
                area = None
            # read in the data
            sio = StringIO(''.join(lines))
            data = pd.read_csv(sio, **keywds)
            names = data.columns.values
            for name in names:
                if name not in units:
                    units[name] = 'unknown'
        # list of properties extracted from the file
        results = [
            pif.Property(
                name=name,
                scalars=list(data[name]),
                units=unit,
                files=pif.FileReference(relative_path=fname),
                methods=pif.Method(name='uniaxial',
                    instruments=pif.Instrument(producer='Mark10')),
                data_type='EXPERIMENTAL',
                tag='Mark10')
            for name,unit in zip(names, units)]
        # Calculate stress from force and cross-sectional area, if provided
        # Both 'area' and 'units' keywords must be given
        if 'force' in names and area is not None:
            # add property to results
            replace_if_present_else_append(results,
                pif.Property(
                    name='area',
                    scalars=area,
                    units=units['area'],
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
                    units=units['stress'],
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
