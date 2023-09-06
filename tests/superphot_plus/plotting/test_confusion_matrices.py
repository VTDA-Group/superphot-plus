import os

from superphot_plus.plotting.confusion_matrices import (
    plot_confusion_matrix,
    plot_high_confidence_confusion_matrix,
    plot_binary_confusion_matrix,
    compare_four_class_confusion_matrices,
    plot_expected_agreement_matrix,
    plot_true_agreement_matrix,
)


def test_plot_confusion_matrices(class_probs_csv, tmp_path):
    """Test functions that plot confusion matrices.
    TODO: add check when one accuracy is 0 (currently divide by zero error).
    """

    # test base plot_confusion_matrix
    y_true_test = [1, 1, 2, 2, 3, 3, 4, 4]
    y_pred_test = [1, 2, 2, 3, 3, 4, 4, 1]
    test_filename_base = os.path.join(tmp_path, "test_cm_base")
    plot_confusion_matrix(y_true_test, y_pred_test, test_filename_base + "_c.pdf", purity=False)
    plot_confusion_matrix(y_true_test, y_pred_test, test_filename_base + "_p.pdf", purity=True)
    assert os.path.exists(test_filename_base + "_c.pdf")
    assert os.path.exists(test_filename_base + "_p.pdf")

    # test plot_high_confidence_confusion_matrix
    test_filename_high_confidence = os.path.join(tmp_path, "test_cm_high_confidence")
    plot_high_confidence_confusion_matrix(class_probs_csv, test_filename_high_confidence)
    assert os.path.exists(test_filename_high_confidence + "_c.pdf")
    assert os.path.exists(test_filename_high_confidence + "_p.pdf")

    # test plot_snIa_confusion_matrix
    test_filename_binary = os.path.join(tmp_path, "test_cm_binary")
    plot_binary_confusion_matrix(class_probs_csv, test_filename_binary)
    assert os.path.exists(test_filename_binary + "_c.pdf")
    assert os.path.exists(test_filename_binary + "_p.pdf")


def test_four_class_cm(class_probs_csv, dummy_alerce_preds, tmp_path):
    """Test generation of both Superphot+ and ALeRCE four-class CMs."""
    compare_four_class_confusion_matrices(class_probs_csv, dummy_alerce_preds, tmp_path, p07=False)
    assert os.path.exists(os.path.join(tmp_path, "superphot4_c.pdf"))
    assert os.path.exists(os.path.join(tmp_path, "alerce_c.pdf"))


def test_agreement_matrix(class_probs_csv, dummy_alerce_preds, tmp_path):
    """Test expected and true agreement matrix generation."""
    plot_expected_agreement_matrix(class_probs_csv, dummy_alerce_preds, tmp_path)
    assert os.path.exists(os.path.join(tmp_path, "expected_agreement.pdf"))

    plot_true_agreement_matrix(class_probs_csv, dummy_alerce_preds, tmp_path, spec=True)
    plot_true_agreement_matrix(class_probs_csv, dummy_alerce_preds, tmp_path, spec=False)

    assert os.path.exists(os.path.join(tmp_path, "true_agreement_spec.pdf"))
    assert os.path.exists(os.path.join(tmp_path, "true_agreement_phot.pdf"))
