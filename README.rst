Citrination Converters
======================

This project provides a central housing for developing converters
from various data sources.

Sources
-------

    1. MTS (Mechanical testing data)
        Reads time, force and displacement data from the CSV output
        written by MTS.

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

```
SOURCE = '{}/data/source_data.csv'.format(HERE)

@pytest.fixture
def equipment_data_no_modifier():
    return

def test_data_no_modifier(equipment_data_no_modifier):
    pifs = pif.dumps(converter(SOURCE), sort_keys=True)
    with open('{}/data/equipment-data-no-modifier.json'.format(HERE), 'w') as ofs:
        ofs.write(pifs)
    assert False
```

2. Check `equipment-data-no-modifier.json` manually.

3. Create a "good" test.

```
SOURCE = '{}/data/source_data.csv'.format(HERE)

@pytest.fixture
def equipment_data_no_modifier():
    with open('{}/data/equipment-data-no-modifier.json'.format(HERE)) as ifs:
        reference = ifs.read()
    return reference

def test_data_no_modifier(equipment_data_no_modifier):
    pifs = pif.dumps(converter(SOURCE), sort_keys=True)

    assert False
```
