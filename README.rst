Citrination Converters
======================

This project provides a central housing for developing converters
from various data sources. *Note* unless otherwise specified, all
keys are processed and stored case insensitively.

Sources
-------

1. Mark10 (Mechanical testing data)
   Reads time, force and displacement data from the CSV output
   written by Mark10. If ``area=(float)``` and ``units=(string)``
   keywords are provided, then the stress will also be calculated.
   The available keys will depend on the input file, but based on
   early testing, are "time", "force", "disp", "area" (if specified),
   "stress" (if ``area`` and ``units`` are specified).

2. Aramis (DIC strain data)
   Reads strain information from CSV output produced by Aramis.

Adding New Tests
----------------

Tests are critical to ensuring the continued stability and performance of
any package. And this one is no exception. At present, the output from the
test is compared against a previous run that was manually inspected. Please
see `test_*.py` for examples. These, however, show the final product. How
do we actually generate those outputs?

1. Run a "bad" test that generates the test file, e.g.
::
    HERE = '/full/path/to/citrine_converters/test'
    SOURCE = '{}/data/source_data.csv'.format(HERE)

    assert os.isdir(HERE)
    assert os.isfile(SOURCE)

    @pytest.fixture
    def equipment_data_no_modifier():
        pass

    def test_data_no_modifier(equipment_data_no_modifier):
        pifs = pif.dumps(converter(SOURCE), sort_keys=True)
        # This will write the output of the test. Be sure to give this
        # a more meaningful name!
        with open('{}/data/equipment-data-no-modifier.json'.format(HERE), 'w') as ofs:
            ofs.write(pifs)
        # The test is intended to fail in this step.
        assert False

2. Check `equipment-data-no-modifier.json` manually.

3. Create a "good" test.
::
    HERE = '/full/path/to/citrine_converters/test'
    SOURCE = '{}/data/source_data.csv'.format(HERE)

    assert os.isdir(HERE)
    assert os.isfile(SOURCE)

    @pytest.fixture
    def equipment_data_no_modifier():
        with open('{}/data/equipment-data-no-modifier.json'.format(HERE)) as ifs:
            reference = ifs.read()
        return reference

    def test_data_no_modifier(equipment_data_no_modifier):
        # These two lines could be combined into a single line,
        # but this is more clear, imho.
        pifs = converter(SOURCE)
        pifs = pif.dumps(pifs, sort_keys=True)
        assert pifs == equipment_data_no_modifier
