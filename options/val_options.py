# Options that are used specifically to configure testing.
# If an options is not set, its default will be used.

from .base_options import BaseOptions


class ValidationOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        self.parser.add_argument('--epoch_count', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')
        self.parser.add_argument('--AtoB', type=bool, default=True, help="If true generate fakes of domain B, if false of domain A")
        self.isTrain = False
