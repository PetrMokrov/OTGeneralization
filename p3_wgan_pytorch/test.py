from p3wgan.experiments import ToyP3WGAN, ToyModelsCreator, QuadraticTransportCostCreator, ToyDistributionCreator
from p3wgan.launcher import ExperimentLauncher

launcher = ExperimentLauncher()

with launcher.localize(device='cpu', num_iter_per_epoch=100):
    launcher.launch(ToyModelsCreator())
    launcher.launch(ToyDistributionCreator())
    launcher.launch(QuadraticTransportCostCreator())
    with launcher.localize(num_epochs=3, draw_images=False):
        launcher.launch(ToyP3WGAN())