'''
Load arguments

Stéphane Vujasinovic
'''

# - IMPORTS ---
from argparse import ArgumentParser


# - FUNCTIONS ---
def args_passer():
    parser = ArgumentParser()
    parser.add_argument('--model', default='./saves/XMem.pth')

    # Dataset Options
    parser.add_argument('--dataset', help='Available dataset name options are : burst-test / burst-val / d16-val / d17-test / d17-val / generic / lvos-test / lvos-val / mose-val / y18-val / y19-val', default='d17-val')
    parser.add_argument('--split', help='val/test', default='val')
    parser.add_argument('--output', default=None)
    parser.add_argument('--save_all', action='store_true',
                        help='Save all frames. Useful only in YouTubeVOS/long-time video', )

    parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')

    # Long-term memory options
    parser.add_argument('--disable_long_term', action='store_true')
    parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
    parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
    parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time',
                                                    type=int, default=10000)
    parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
    parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)

    # Multi-scale options
    parser.add_argument('--save_scores', action='store_true',
                        help="Save the probabilities as well as the logits.")
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--size', default=480, type=int,
                help='Resize the shorter side to this size. -1 to use original resolution.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for the dataloader')
    parser.add_argument('--verbose', action='store_false', help='Disabeling icecream prints')
    parser.add_argument('--iouatX', type=float, default=0.0, help="Issue a prompt if IoU belove this value")
    parser.add_argument('--HatX', type=float, default=1000.0, help="Issue a prompt if entropy belove this value")
    parser.add_argument('--derivatX', type=float, default=1000.0, help="Issue a prompt is the derivative is above this value")
    parser.add_argument('--working_upd', action='store_true', help='Update the WORKING memory with the resulted prompted mask')
    parser.add_argument('--deep_upd', action='store_true', help="Update the Deeper memory with the resulted prompted mask")
    parser.add_argument('--save_prompts', action='store_true', help="Save the prompts only if used")
    parser.add_argument('-T', '--temperature', default=1.0, help="Apply temperature scaling")
    parser.add_argument('--Mem_upd', type=float, default=1000.0, help="Issue a prompt if entropy belove this value")
    parser.add_argument('--update_mem_with_prompt', action='store_true', help="Whenever a prompt is called, update the memory accordingly")
    parser.add_argument('--debug', action='store_true', help="debugging")
    parser.add_argument('--pseudo_entropy_seuil', type=float, default=1000.0, help="Issue a prompt if entropy belove this value")
    parser.add_argument('--pseudo_derivatX', type=float, default=1000.0, help="Issue a prompt is the derivative is above this value")
    parser.add_argument('--update_mem_with_pseudo_prompt', action='store_true', help="Whenever a prompt is called, update the memory accordingly")
    parser.add_argument('--mode', type=int, default=0, help="0: to take the XMem mask, 1: to take the weighted XMem mask with the confidence")
    parser.add_argument('--using_mask_in_pseudo', action='store_true', help="Should ituse the original XMem mask for its prediction?")
    parser.add_argument('--user_pseudo_BBox', action='store_true', help="Should it use BBox as pseudo interactions instead of clicks?")
    parser.add_argument('--use_negative_clicks', action='store_true', help="Include negative interactions based for the user?")
    args = parser.parse_args()
    
    return args