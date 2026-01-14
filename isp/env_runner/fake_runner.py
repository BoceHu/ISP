from isp.env_runner.base_image_runner import BaseImageRunner
from isp.policy.base_image_policy import BaseImagePolicy


class FakeRunner(BaseImageRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BaseImagePolicy):
        return dict()
