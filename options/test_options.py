# Options that are used specifically to configure testing.
# If an options is not set, its default will be used.

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--model_path', type=str, help='path of the pretrained model')
        self.parser.add_argument('--num_test', type=int, default=1, help='Number of batches to process')
        self.parser.add_argument('--num_visuals', type=int, default=0, help='Number of samples to generate images from')
        self.parser.add_argument('--visuals', action='store_true', help='Number of batches to process')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.isTrain = False
