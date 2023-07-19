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
        mapping an integer classification to the type string.

        Parameters
        ----------
        cls : SupernovaClass
            The SupernovaClass class.
        allowed_types : list of str, optional
            The list of supernova classes to be included in the mapping.
            Defaults to ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"].

        Returns
        -------
        tuple of dict
            Tuple containing the mapping of string labels to integer identifiers
            and the mapping of integer identifiers to string labels, respectively.
        """
        allowed_types = [a_type.value if isinstance(a_type, cls) else a_type for a_type in allowed_types]

        labels_to_classes = {a_type: i for i, a_type in enumerate(allowed_types)}
        classes_to_labels = dict(enumerate(allowed_types))

        return labels_to_classes, classes_to_labels

    @classmethod
    def get_alts(cls):
        """Returns the alternative namings for each supernova class.

        Parameters
        ----------
        cls : SupernovaClass
            The SupernovaClass class.

        Returns
        -------
        dict
            A dictionary that maps each supernova class label to its
            respective alternative namings.
        """
        return {
            cls.SUPERNOVA_IA.value: [
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
            cls.SUPERNOVA_IBC.value: [
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
            cls.SUPERNOVA_IIN.value: ["SNIIn", "35", "SLSN-II"],
            cls.SUPERLUMINOUS_SUPERNOVA_I.value: ["40", "SLSN"],
            cls.SUPERLUMINOUS_SUPERNOVA_II.value: ["SN IIP", "SN IIL", "SNII", "SNIIP", "32", "30", "31"],
            "TDE": ["42"],
        }

    @classmethod
    def get_reflect_style(cls, label):
        """Returns a supernova class label in reflect style.

        Parameters
        ----------
        cls : SupernovaClass
            The SupernovaClass class.
        label : str
            The predicted supernova class label, provided by the
            ZTF classifier, for which the reflected style is requested.

        Returns
        -------
        str
            The supernova class label in reflect style.
        """
        label_reflect_style = {
            "SNII": cls.SUPERNOVA_II.value,
            "SNIa": cls.SUPERNOVA_IA.value,
            "SLSN": cls.SUPERLUMINOUS_SUPERNOVA_I.value,
            "SNIbc": cls.SUPERNOVA_IBC.value,
        }
        return label_reflect_style[label]
