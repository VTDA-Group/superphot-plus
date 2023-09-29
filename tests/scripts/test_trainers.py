import os
import tempfile

import numpy as np
import pytest

from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.supernova_properties import SupernovaProperties
from superphot_plus.trainers.classifier_trainer import ClassifierTrainer
from superphot_plus.trainers.mosfit_trainer import MosfitTrainer
from superphot_plus.tuners.classifier_tuner import ClassifierTuner
from superphot_plus.tuners.mosfit_tuner import MosfitTuner


@pytest.mark.asyncio
async def test_classifier():
    """Generates mock posterior samples and invokes classifier tuning and training."""
    input_dim = 15
    num_samples = 100
    goal_per_class = 20  # There are 5 supernova classes

    # Generate mock data
    names = np.array([f"ZTF_{str(n).zfill(6)}" for n in range(num_samples)])
    labels = np.array([[s_class] * goal_per_class for s_class in SnClass.all_classes()]).flatten()
    redshifts = np.random.uniform(0, 1, size=num_samples).flatten()
    posteriors = {name: np.random.random((num_samples, input_dim)) for name in names}

    data = (names, labels, redshifts, posteriors)

    # Train mosfit models for each physical property
    with tempfile.TemporaryDirectory() as tmp_dir:
        await run_single_classifier(data, tmp_dir)


async def run_single_classifier(data, tmp_dir):
    """
    Tests that we can tune the classifier using K-Fold cross validation to
    obtain a best hyperparameter set. It also tests that we are able to train the
    model and run evaluation by obtaining predictions on the test holdout set.
    """
    # First, run classifier tuner using K-Fold CV
    ClassifierTuner(
        sampler="dynesty",
        classification_dir=tmp_dir,
    ).run(data=data, num_hp_samples=1)

    # Then, train and evaluate the model
    ClassifierTrainer(
        config_name="best-config",
        sampler="dynesty",
        classification_dir=tmp_dir,
    ).run(data=data)

    # Make sure model was generated
    assert os.path.exists(os.path.join(tmp_dir, "models", "best-config.yaml"))
    assert os.path.exists(os.path.join(tmp_dir, "models", "best-model.pt"))

    # Make sure logs and predictions on the test set were written to disk
    assert os.path.exists(os.path.join(tmp_dir, "logs.txt"))
    assert os.path.exists(os.path.join(tmp_dir, "probs.csv"))


@pytest.mark.asyncio
async def test_mosfit():
    """
    Generates mock posterior samples and physical property data and
    invokes mosfit model tuning and training.
    """
    input_dim = 15
    num_samples = 100

    # Generate mock data
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

    data = (names, posteriors, properties)

    # Train mosfit models for each physical property
    with tempfile.TemporaryDirectory() as tmp_dir:
        for parameter in SupernovaProperties.all_properties():
            await run_single_mosfit(data, parameter, tmp_dir)


async def run_single_mosfit(data, parameter, tmp_dir):
    """
    Tests that we can tune the MOSFiT regressor using K-Fold cross validation to
    obtain a best hyperparameter set. It also tests that we are able to train the
    model and run evaluation by obtaining predictions on the test holdout set.
    """
    # First, run mosfit tuner using K-Fold CV
    MosfitTuner(
        parameter=parameter,
        sampler="svi",
        mosfit_dir=tmp_dir,
    ).run(data=data, num_hp_samples=1)

    # Then, train and evaluate the model
    MosfitTrainer(
        parameter=parameter,
        sampler="svi",
        mosfit_dir=tmp_dir,
    ).run(data=data)

    # Make sure model was generated
    assert os.path.exists(os.path.join(tmp_dir, "models", f"{parameter}.yaml"))
    assert os.path.exists(os.path.join(tmp_dir, "models", f"{parameter}.pt"))

    # Make sure logs and predictions on the test set were written to disk
    assert os.path.exists(os.path.join(tmp_dir, f"{parameter}_logs.txt"))
    assert os.path.exists(os.path.join(tmp_dir, f"{parameter}_preds.csv"))
