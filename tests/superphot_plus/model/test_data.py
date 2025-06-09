from snapi import SamplerResultGroup


def test_oversample_using_posteriors(test_sampler_results, single_ztf_sn_id):
    """Test oversampling using posteriors"""

    names = [single_ztf_sn_id] * 3
    samples_per_majority_class = 10
    redshifts = [4.5, 4.5, -1]

    # Oversampling from a set of unique supernova classes.
    classes = [4, 1, 2]  # Classes for "Sn Ibc", "SN II" and "SN IIn"
    
    srg = SamplerResultGroup.load(
        test_sampler_results
    )
    # oversample using trainer
    # TODO!!!

    # We should have 30 samples in total, 10 for each class.
    assert len(features) == len(labels) == 30
    assert len(labels[labels == 4]) == len(labels[labels == 1]) == len(labels[labels == 2]) == 10

    # add redshifts and oversample here!!!
    # TODO!!!

    # We should have 30 samples in total, 10 for each class.
    assert len(features) == len(labels) == 20
    assert len(labels[labels == 4]) == len(labels[labels == 1]) == 10
    assert len(labels[labels == 2]) == 0
    
    # Oversampling from a set with repeated supernova classes.
    # Modify labels here and oversample
    # TODO!!!

    # Due to repeated class, we draw 2*10 PER CLASS
    assert len(features) == len(labels) == 40
    assert len(labels[labels == 4]) == len(labels[labels == 1]) == 20

