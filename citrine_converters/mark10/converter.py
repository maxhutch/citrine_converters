# -*- coding: utf-8 -*-

from pypif import pif
import pandas as pd
from ..tools import replace_if_present_else_append


def converter(files=[], **keywds):
    """
    Summary
    =======

    Converter to calculate stress data from Mark10 CSV output.

    Input
    =====
    :files, list: List of CSV-formatted files.

    Options
    -------
    :area, float: Cross sectional area of the sample.
    :units, string: Area units.

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
            # path is currently discarded -- include in metadata store?
            path = ifs.readline().strip()
            # ultimately names are used to name the columns.
            # render case insensitive (lowercase)
            names = [entry.strip().lower()
                     for entry in ifs.readline().split(',')]
            # units are currently discarded as well, but these, too,
            # should be included in the metadata store.
            units = [entry.strip().strip('()')
                     for entry in ifs.readline().split(',')]
            # read in the data
            data = pd.read_csv(ifs, names=names)
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
        if 'force' in names and 'area' in keywds and 'units' in keywds:
            area = float(keywds['area'])
            stress_units = '{}/{}'.format(dict(zip(names, units))['force'],
                                          keywds['units'])
            # add property to results
            replace_if_present_else_append(results,
                pif.Property(
                    name='area',
                    scalars=area,
                    units=keywds['units'],
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
