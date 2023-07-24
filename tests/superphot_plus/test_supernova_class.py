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
    allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc", "SLSN-II"]

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
        SnClass.SUPERLUMINOUS_SUPERNOVA_II,
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


def test_supernovae_alternatives():
    """Test that we get alternative namings for each supernova class."""
    expected_alts = {
        SnClass.SUPERNOVA_IA.value: [
            "SN Ia-91T-like",
            "SN Ia-CSM",
            "SN Ia-91bg-like",
            "SNIa",
            "SN Ia-91T",
            "SN Ia-91bg",
            "10",
            "11",
            "12",
        ],
        SnClass.SUPERNOVA_IBC.value: [
            "SN Ic",
            "SN Ib",
            "SN Ic-BL",
            "SN Ib-Ca-rich",
            "SN Ib/c",
            "SNIb",
            "SNIc",
            "SNIc-BL",
            "21",
            "20",
            "27",
            "26",
            "25",
        ],
        SnClass.SUPERNOVA_IIN.value: ["SNIIn", "35", "SLSN-II"],
        SnClass.SUPERLUMINOUS_SUPERNOVA_I.value: ["40", "SLSN"],
        SnClass.SUPERLUMINOUS_SUPERNOVA_II.value: ["SN IIP", "SN IIL", "SNII", "SNIIP", "32", "30", "31"],
        "TDE": ["42"],
    }

    assert SnClass.get_alternative_namings() == expected_alts


def test_reflect_style():
    """Test that we get the reflect style naming for each supernova class."""
    assert SnClass.get_reflect_style("SNII") == SnClass.SUPERNOVA_II.value
    assert SnClass.get_reflect_style("SNIa") == SnClass.SUPERNOVA_IA.value
    assert SnClass.get_reflect_style("SLSN") == SnClass.SUPERLUMINOUS_SUPERNOVA_I.value
    assert SnClass.get_reflect_style("SNIbc") == SnClass.SUPERNOVA_IBC.value
