# -*- coding: utf-8 -*-

from pypif import pif
import re
import pandas as pd

def converter(files=[], **keywds):
    """
    Summary
    =======

    Converter to calculate stress data from MTS CSV output.

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
            # column names from Aramis are not well organized (IMHO):
            # refactor
            names = [entry.strip().lower()
                     for entry in ifs.readline().split(',')]
            # `label` format: DESCRIPTION (REDUCTION): LABEL [UNITS]
            # desired format: LABEL (UNITS)
            label = names[1]
            fmtstr = r'[^(]+\(([^)]+)\):\s*([^[]+)\[([^]]+)\]'
            try:
                reduction, label, units = re.search(fmtstr, label).groups()
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
                data_type='EXPERIMENTAL',
                tag=reduction)
            for name,unit in zip(names, units)]
        # Determine the time at which each measurement was taken
        if 'timestep' in keywds:
            #+ ensure timestep is a float
            timestep = float(keywds['timestep'])
            results.append(pif.Property(
                name='time',
                scalars=list(data[names[0]]*timestep),
                units='s',
                data_type='EXPERIMENTAL',
                tag=reduction))
    # job's done!
    return results
