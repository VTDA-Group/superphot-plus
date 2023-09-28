from superphot_plus.samplers.flowmc_sampler import FlowMCSampler


def test_flow_mc_single_file(single_ztf_lightcurve_object, ztf_priors):
    """Just test that we generated a new file with fits"""
    sampler = FlowMCSampler()
    posterior_samples = sampler.run_single_curve(
        single_ztf_lightcurve_object,
        priors=ztf_priors,
        rng_seed=4,
        n_chains=1,
        num_epochs=1,
    )

    sample_mean = posterior_samples.sample_mean()
    assert len(sample_mean) == 14
    assert len(posterior_samples.samples) == 100
