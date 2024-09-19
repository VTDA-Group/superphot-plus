from superphot_plus.model.data import PosteriorSamplesGroup
from superphot_plus.utils import retrieve_posterior_set


def test_oversample_using_posteriors(test_data_dir, single_ztf_sn_id):
    """Test oversampling using posteriors"""

    names = [single_ztf_sn_id] * 3
    samples_per_majority_class = 10
    redshifts = [4.5, 4.5, -1]

    # Oversampling from a set of unique supernova classes.
    classes = [4, 1, 2]  # Classes for "Sn Ibc", "SN II" and "SN IIn"
    
    all_post_objs = retrieve_posterior_set(
        names,
        test_data_dir,
        sampler='dynesty',
        redshifts=None, # ignore redshift constraints
        labels=classes
    )
    psg = PosteriorSamplesGroup(
        all_post_objs,
        use_redshift_info=False
    )
    features, labels = psg.oversample(samples_per_majority_class)

    # We should have 30 samples in total, 10 for each class.
    assert len(features) == len(labels) == 30
    assert len(labels[labels == 4]) == len(labels[labels == 1]) == len(labels[labels == 2]) == 10

    all_post_objs2 = retrieve_posterior_set(
        names,
        test_data_dir,
        sampler='dynesty',
        redshifts=redshifts,
        labels=classes
    )
    psg2 = PosteriorSamplesGroup(
        all_post_objs2,
        use_redshift_info=True
    )
    features, labels = psg2.oversample(samples_per_majority_class)

    # We should have 30 samples in total, 10 for each class.
    assert len(features) == len(labels) == 20
    assert len(labels[labels == 4]) == len(labels[labels == 1]) == 10
    assert len(labels[labels == 2]) == 0
    
    # Oversampling from a set with repeated supernova classes.
    psg.labels = [4, 1, 1]  # Classes for "Sn Ibc" and "SN II"
    features, labels = psg.oversample(samples_per_majority_class)

    # Due to repeated class, we draw 2*10 PER CLASS
    assert len(features) == len(labels) == 40
    assert len(labels[labels == 4]) == len(labels[labels == 1]) == 20

