"""
Argument loaders for XMem.

by Stéphane Vujasinovic
"""
# - IMPORTS ---
from argparse import ArgumentParser

# - FUNCTIONS ---
class BaseArgParser():
    def __init__(self):
        self.parser = ArgumentParser()
        
        self.parser.add_argument('--model', default='./saves/XMem.pth')

        # Data options
        self.parser.add_argument('--d16_path', default='../DAVIS/2016')
        self.parser.add_argument('--d17_path', default='../DAVIS/2017')
        self.parser.add_argument('--y18_path', default='../YouTube2018')
        self.parser.add_argument('--y19_path', default='../YouTube')
        self.parser.add_argument('--lv_path', default='../long_video_set')
        self.parser.add_argument('--burst_path', default='../BURST')
        self.parser.add_argument('--vot_path', default='../VOTS2023')
        self.parser.add_argument('--mose_path', default='../MOSE')
        # For generic (G) evaluation, point to a folder that contains "JPEGImages" and "Annotations"
        self.parser.add_argument('--generic_path')

        self.parser.add_argument('--dataset', help='Available dataset name options are : burst-test / burst-val / d16-val / d17-test / d17-val / generic / lvos-test / lvos-val / mose-val / y18-val / y19-val', default='d17-val')

        self.parser.add_argument('--split', help='val/test', default='val')
        self.parser.add_argument('--output', default=None)
        self.parser.add_argument('--save_all', action='store_true',
                    help='Save all frames. Useful only in YouTubeVOS/long-time video', )

        self.parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')

        # Long-term memory options
        self.parser.add_argument('--disable_long_term', action='store_true')
        self.parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
        self.parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
        self.parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time',
                                                        type=int, default=10000)
        self.parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

        self.parser.add_argument('--top_k', type=int, default=30)
        self.parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
        self.parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)

        # Multi-scale options
        self.parser.add_argument('--save_scores', action='store_true',
                            help="Save the probabilities as well as the logits.")
        self.parser.add_argument('--flip', action='store_true')
        self.parser.add_argument('--size', default=480, type=int,
                    help='Resize the shorter side to this size. -1 to use original resolution.')
        self.parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for the dataloader')
        self.parser.add_argument('--verbose', action='store_false', help='Disabeling icecream prints')
        
    def arguments_parser(self):
        # Load/Set arguments
        return self.parser.parse_args()
    
    
class UXMemArgParser(BaseArgParser):
    def __init__(self):
        super().__init__()
        
        # Give additional arguments
