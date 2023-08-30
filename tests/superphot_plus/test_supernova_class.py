import numpy as np
import pytest

from superphot_plus.supernova_class import SupernovaClass as SnClass


def test_default_type_maps():
    """Test that we can get the default supernovae type maps."""
    expected_classes = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]

    labels_to_classes, classes_to_labels = SnClass.get_type_maps()

    assert list(labels_to_classes.keys()) == expected_classes
    assert list(classes_to_labels.values()) == expected_classes
    assert list(labels_to_classes.values()) == list(classes_to_labels.keys())


def test_type_maps_for_allowed_types_string():
    """Test that we can get the supernovae mappings for allowed string types."""
    allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]

    labels_to_classes, classes_to_labels = SnClass.get_type_maps(allowed_types)

    assert list(labels_to_classes.keys()) == allowed_types
    assert list(classes_to_labels.values()) == allowed_types
    assert list(labels_to_classes.values()) == list(classes_to_labels.keys())


def test_type_maps_for_allowed_types_enum():
    """Test that we can get the supernovae mappings for allowed types
    of the SupernovaClass enum."""
    allowed_types = [
        SnClass.SUPERNOVA_IA,
        SnClass.SUPERNOVA_IBC,
        SnClass.SUPERNOVA_II,
        SnClass.SUPERNOVA_IIN,
        SnClass.SUPERLUMINOUS_SUPERNOVA_I,
        # SnClass.SUPERLUMINOUS_SUPERNOVA_II,
    ]

    labels_to_classes, classes_to_labels = SnClass.get_type_maps(allowed_types)

    assert list(labels_to_classes.keys()) == allowed_types
    assert list(classes_to_labels.values()) == allowed_types
    assert list(labels_to_classes.values()) == list(classes_to_labels.keys())


def test_type_maps_for_allowed_types_mixed():
    """Test that we can get the supernovae mappings for mixed allowed types
    (of string, enum or a mix of both)."""
    allowed_types = [
        SnClass.SUPERNOVA_IA,
        "SN II",
        SnClass.SUPERNOVA_IIN,
        SnClass.SUPERLUMINOUS_SUPERNOVA_I,
        "SN Ibc",
        "SLSN-II",
    ]

    labels_to_classes, classes_to_labels = SnClass.get_type_maps(allowed_types)

    assert list(labels_to_classes.keys()) == allowed_types
    assert list(classes_to_labels.values()) == allowed_types
    assert list(labels_to_classes.values()) == list(classes_to_labels.keys())


def test_canonicalize():
    """Test that we can canonicalize all the labels and alternative labels."""

    for sn_class in SnClass:
        canon = SnClass.canonicalize(sn_class.value)
        assert canon == sn_class.value

    alts = SnClass.get_alternative_namings()

    for alt_canon, alt_others in alts.items():
        for other in alt_others:
            canon = SnClass.canonicalize(other)
            assert canon == alt_canon

    canon = SnClass.canonicalize("nonsense")
    assert canon == "nonsense"


def test_alerce_to_superphot():
    """Test that we convert from ALeRCE to Superphot+ naming for each supernova class."""
    assert SnClass.from_alerce_to_superphot_format("SNII") == SnClass.SUPERNOVA_II.value
    assert SnClass.from_alerce_to_superphot_format("SNIa") == SnClass.SUPERNOVA_IA.value
    assert SnClass.from_alerce_to_superphot_format("SLSN") == SnClass.SUPERLUMINOUS_SUPERNOVA_I.value
    assert SnClass.from_alerce_to_superphot_format("SNIbc") == SnClass.SUPERNOVA_IBC.value


def test_get_classes_from_labels():
    """Test that we can get a list of classes from a list of supernova labels."""
    classes = SnClass.get_classes_from_labels(
        [SnClass.SUPERNOVA_IA, SnClass.SUPERNOVA_II, SnClass.SUPERNOVA_IIN]
    )

    # Array values are of type int
    assert classes.dtype.type is np.int_

    # When label is invalid, an exception is thrown.
    with pytest.raises(ValueError):
        SnClass.get_classes_from_labels(["TEST"])


def test_get_labels_from_classes():
    """Test that we can get a list of supernova labels from a list of classes."""
    labels = SnClass.get_labels_from_classes([0, 1, 2])

    # Array values are of type string
    assert labels.dtype.type is np.str_

    # When class is invalid, an exception is thrown.
    with pytest.raises(ValueError):
        SnClass.get_labels_from_classes([0, 1, 100])
