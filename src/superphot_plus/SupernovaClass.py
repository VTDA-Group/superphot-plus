from enum import Enum


class SupernovaClass(str, Enum):
    """Classes of Supernovae"""

    SUPERNOVA_IA = "SN Ia"
    SUPERNOVA_IBC = "SN Ibc"
    SUPERNOVA_II = "SN II"
    SUPERNOVA_IIN = "SN IIn"
    SUPERLUMINOUS_SUPERNOVA_I = "SLSN-I"
    SUPERLUMINOUS_SUPERNOVA_II = "SLSN-II"

    @classmethod
    def get_type_maps(cls, allowed_types=["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]):
        """For some allowed supernova classes, create dictionaries
        mapping an integer classification to the type string."""
        allowed_types = [a_type.value if isinstance(a_type, cls) else a_type for a_type in allowed_types]

        labels_to_classes = {a_type: i for i, a_type in enumerate(allowed_types)}
        classes_to_labels = dict(enumerate(allowed_types))

        return labels_to_classes, classes_to_labels

    @classmethod
    def get_allowed_types(cls):
        """Returns all supernova class types as strings."""
        return list(map(lambda sn_class: sn_class.value, cls))

    @classmethod
    def get_alts(cls, sn_type):
        """Returns the alternative namings for each supernova class."""
        alts = []
        match sn_type:
            case cls.SUPERNOVA_IA:
                alts = [
                    "SN Ia-91T-like",
                    "SN Ia-CSM",
                    "SN Ia-91bg-like",
                    "SNIa",
                    "SN Ia-91T",
                    "SN Ia-91bg",
                    "10",
                    "11",
                    "12",
                ]
            case cls.SUPERNOVA_IBC:
                alts = [
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
                ]
            case cls.SUPERNOVA_IIN:
                alts = ["SNIIn", "35", "SLSN-II"]
            case cls.SUPERLUMINOUS_SUPERNOVA_I:
                alts = ["40", "SLSN"]
            case cls.SUPERLUMINOUS_SUPERNOVA_II:
                alts = ["SN IIP", "SN IIL", "SNII", "SNIIP", "32", "30", "31"]
        return alts
