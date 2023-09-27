import os
import tempfile

import numpy as np
import pytest

from superphot_plus.supernova_properties import SupernovaProperties
from superphot_plus.trainers.mosfit_trainer import MosfitTrainer
from superphot_plus.tuners.mosfit_tuner import MosfitTuner


@pytest.mark.asyncio
async def test_mosfit():
    """
    Generates mock posterior samples and physical property data and
    invokes mosfit model tuning and training.
    """
    # Create mock input data
    input_dim = 15
    num_samples = 100

    names = np.array([f"ZTF_{str(n).zfill(6)}" for n in range(num_samples)])
    posteriors = np.random.random((num_samples, input_dim))
    properties = np.array(
        [
            SupernovaProperties(
                bfield=np.random.uniform(),
                pspin=np.random.uniform(),
                mejecta=np.random.uniform(),
                vejecta=np.random.uniform(),
            )
            for _ in range(num_samples)
        ]
    )

    # Train mosfit models for each physical property
    with tempfile.TemporaryDirectory() as tmp_dir:
        for parameter in SupernovaProperties.all_properties():
            await run_single_mosfit(names, posteriors, properties, parameter, tmp_dir)


async def run_single_mosfit(names, posteriors, properties, parameter, tmp_dir):
    """
    Tests that we can tune the MOSFiT regressor using K-Fold cross validation to
    obtain a best hyperparameter set. It also tests that we are able to train the
    model and run evaluation by obtaining predictions on the test holdout set.
    """
    # First, run mosfit tuner using K-Fold CV
    tuner = MosfitTuner(parameter=parameter, sampler="svi", mosfit_dir=tmp_dir)
    tuner.run(data=(names, posteriors, properties), num_hp_samples=1)

    # Then, train and evaluate the model
    trainer = MosfitTrainer(parameter=parameter, sampler="svi", mosfit_dir=tmp_dir)
    trainer.run(data=(names, posteriors, properties))

    # Make sure model was generated
    assert os.path.exists(os.path.join(tmp_dir, "models", f"{parameter}.yaml"))
    assert os.path.exists(os.path.join(tmp_dir, "models", f"{parameter}.pt"))

    # Make sure logs and predictions on the test set were written to disk
    assert os.path.exists(os.path.join(tmp_dir, f"{parameter}_logs.txt"))
    assert os.path.exists(os.path.join(tmp_dir, f"{parameter}_preds.csv"))
