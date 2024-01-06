import os
import tempfile

from generate_fits import PosteriorsGenerator

def test_generate_fits():
    """Tests posteriors generation for a variety of
    samplers using a fixed seed."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        for sampler in [
            "dynesty",
            "svi",
            # "NUTS",
            "iminuit",
            "licu-ceres",
            "licu-mcmc-ceres",
        ]:
            lightcurves_dir = "tests/data/ztf_lcs"
            fits_dir = os.path.join(tmp_dir, f"{sampler}_fits")

            PosteriorsGenerator(
                sampler_name=sampler,
                lightcurves_dir=lightcurves_dir,
                survey="ZTF",
                num_workers=1,
                output_dir=tmp_dir,
            ).generate_data(seed=4)

            assert os.path.exists(fits_dir)
            assert len(os.listdir(fits_dir)) == len(os.listdir(lightcurves_dir))
