# -*- coding: utf-8 -*-

from pypif import pif
import re
import numpy as np
import pandas as pd

def converter(files=[], **keywds):
    """
    Summary
    =======

    Converter to calculate strain data from Aramis CSV output.

    Input
    =====
    :files, str or list: One or list of CSV-formatted files.

    Options
    -------
    :timestep, float: Interval (seconds) with which strain data is collected.

    Output
    ======
    PIF object or list of PIF objects
    """
    # Handle required input parameters
    #+ ensure *files* is a list. If a single file is passed, convert it
    #+ into a list.
    if isinstance(files, str):
        files = [files]
    # Process filenames
    for fname in files:
        with open(fname) as ifs:
            # defaultcode/encoding is currently discarded
            junk = ifs.readline()
            # "Statistics export" line is currently discarded
            junk = ifs.readline()
            # refactor column names from Aramis
            names = [entry.strip().lower()
                     for entry in ifs.readline().split(',')]
            #+ names[0] (strain stage): no change
            #
            #+ names[1] (strain)
            # `label` format: DESCRIPTION (REDUCTION): LABEL [UNITS]
            # desired format: LABEL (UNITS)
            label = names[1]
            fmark10tr = r'[^(]+\(([^)]+)\):\s*([^[]+)\[([^]]+)\]'
            try:
                reduction, label, units = re.search(fmark10tr, label).groups()
                names[1] = label
            except ValueError:
                msg = '"{}" in {} is not a valid label format.'.format(
                    label, fname)
                raise ValueError(msg)
            # restructure names and units
            names = [names[0], label.strip()]
            units = ['None', units]
            # read in the data
            data = pd.read_csv(ifs, names=names)
        # list of properties extracted from the file
        results = [
            pif.Property(
                name=name,
                scalars=list(data[name]),
                units=unit,
                files=pif.FileReference(relative_path=fname),
                methods=pif.Method(name='digital image correlation (DIC)',
                    instruments=pif.Instrument(name='DIC', producer='Aramis')),
                data_type='EXPERIMENTAL',
                tag=reduction)
            for name,unit in zip(names, units)]
        # strain (results[1]) transforms
        strain = results[1]
        #+ standardize naming convention
        strain_type = strain.name
        strain.name = 'strain'
        try:
            strain.tag.append(strain_type)
        except AttributeError:
            strain.tag = [strain.tag, strain_type]
        #+ is a transform from % strain necessary?
        if strain.units == '%':
            strain.scalars = list(np.divide(strain.scalars, 100.))
            strain.units = 'mm/mm'
        # Determine the time at which each measurement was taken
        if 'timestep' in keywds:
            #+ ensure timestep is a float
            timestep = float(keywds['timestep'])
            results.append(pif.Property(
                name='time',
                scalars=list(data[names[0]]*timestep),
                units='s',
                files=pif.FileReference(relative_path=fname),
                methods=pif.Method(name='digital image correlation (DIC)',
                    instruments=pif.Instrument(name='DIC', producer='Aramis')),
                data_type='EXPERIMENTAL',
                tag=reduction))
    # Wrap in system object
    results = pif.System(
        names='Aramis',
        properties=results,
        tags=files)
    # job's done!
    return results
