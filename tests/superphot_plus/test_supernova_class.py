from superphot_plus.supernova_class import SupernovaClass as SnClass


def test_get_type_maps():
    """Test that we can get the default supernovae type maps."""
    expected_classes = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]

    labels_to_classes, classes_to_labels = SnClass.get_type_maps()

    assert list(labels_to_classes.keys()) == expected_classes
    assert list(classes_to_labels.values()) == expected_classes


def test_get_alts():
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

    alts = SnClass.get_alts()

    assert alts == expected_alts


def test_get_reflect_style():
    """Test that we get the reflect style naming for each supernova class."""
    reflect_style_II = SnClass.get_reflect_style("SNII")
    reflect_style_Ia = SnClass.get_reflect_style("SNIa")
    reflect_style_SLSN = SnClass.get_reflect_style("SLSN")
    reflect_style_SNIbc = SnClass.get_reflect_style("SNIbc")

    assert reflect_style_II == SnClass.SUPERNOVA_II.value
    assert reflect_style_Ia == SnClass.SUPERNOVA_IA.value
    assert reflect_style_SLSN == SnClass.SUPERLUMINOUS_SUPERNOVA_I.value
    assert reflect_style_SNIbc == SnClass.SUPERNOVA_IBC.value
