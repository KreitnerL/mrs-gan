# Options that are used specifically to configure testing.
# If an options is not set, its default will be used.

from .base_options import BaseOptions


class ValidationOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--model_path', type=str, help='path of the pretrained model')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        self.isTrain = False
