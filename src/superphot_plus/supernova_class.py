from enum import Enum

import numpy as np


class SupernovaClass(str, Enum):
    """Classes of Supernovae"""

    SUPERNOVA_IA = "SN Ia"
    SUPERNOVA_IBC = "SN Ibc"
    SUPERNOVA_II = "SN II"
    SUPERNOVA_IIN = "SN IIn"
    SUPERLUMINOUS_SUPERNOVA_I = "SLSN-I"
    SUPERNOVA_IBN = "SN Ibn"

    @classmethod
    def get_type_maps(cls, allowed_types=None):
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
        if not allowed_types:
            allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]

        allowed_types = [a_type.value if isinstance(a_type, cls) else a_type for a_type in allowed_types]

        labels_to_classes = {a_type: i for i, a_type in enumerate(allowed_types)}
        classes_to_labels = dict(enumerate(allowed_types))

        return labels_to_classes, classes_to_labels

    @classmethod
    def get_alternative_namings(cls):
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
                "SNIbc",
                "SNIc-BL",
                "21",
                "20",
                "27",
                "26",
                "25",
            ],
            cls.SUPERNOVA_II.value: ["SN IIP", "SN IIL", "SNII", "SNIIP", "32", "30", "31"],
            cls.SUPERNOVA_IIN.value: ["SNIIn", "35", "SLSN-II", "SLSNII"],
            cls.SUPERLUMINOUS_SUPERNOVA_I.value: ["40", "SLSN"],
            cls.SUPERNOVA_IBN.value: ["SN Ibn", "SNIbn", "SNIcn", "SN Icn", "SN Ibn/Icn"]
            # cls.SUPERLUMINOUS_SUPERNOVA_II.value: [],
            #"TDE": ["TDE", "42"],
        }

    @classmethod
    def canonicalize(cls, label):
        """Returns a canonical label, using the proper and alternative namings for
        each supernova class.

        Parameters
        ----------
        cls : SupernovaClass
            The SupernovaClass class.
        label : str
            The label to canonicalize

        Returns
        -------
        str
            original label if already canonical, supernova class string if found in
            dictionary of alternative names, or the original label if not able to be
            canonicalized.
        """
        if label in cls.all_classes():
            return label

        alts = cls.get_alternative_namings()
        for canon_label, other_names in alts.items():
            if label in other_names:
                return canon_label
        return label

    @classmethod
    def from_alerce_to_superphot_format(cls, label):
        """Converts label from ALeRCE's to Superphot+'s formatting.

        Parameters
        ----------
        cls : SupernovaClass
            The SupernovaClass class.
        label : str
            The predicted supernova class label, provided by the
            ZTF classifier, but in ALeRCE format.

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

    @classmethod
    def all_classes(cls):
        """Fetch a list of all supernova classes"""
        return [t.value for t in cls]

    @classmethod
    def get_classes_from_labels(cls, labels):
        """Returns the integer classes for a list of supernova labels.

        Parameters
        ----------
        cls : SupernovaClass
            The SupernovaClass class.
        labels : list of str
            The supernovae string labels.

        Returns
        -------
        np.array of int
            Numpy array containing the supernovae integer classes.
        """
        labels_to_classes, _ = cls.get_type_maps()
        if any(l not in labels_to_classes for l in labels):
            raise ValueError("Invalid supernova label.")
        return np.array([labels_to_classes[l] for l in labels]).astype(int)

    @classmethod
    def get_labels_from_classes(cls, classes):
        """Returns the supernova labels for a list of integer classes.

        Parameters
        ----------
        cls : SupernovaClass
            The SupernovaClass class.
        classes : list of int
            The supernovae integer classes.

        Returns
        -------
        np.array of str
            Numpy array containing the supernovae string labels.
        """
        _, classes_to_labels = cls.get_type_maps()
        if any(c not in classes_to_labels for c in classes):
            raise ValueError("Invalid supernova class.")
        return np.array([classes_to_labels[c] for c in classes]).astype(str)
