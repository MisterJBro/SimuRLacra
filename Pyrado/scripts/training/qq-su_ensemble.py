"""
Use an ensemble of trained policies to solve the qq-su environment.
"""
import torch as to
from multiprocessing import freeze_support

import pyrado
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.utils.data_types import RenderMode
from pyrado.policies.special.ensemble import EnsemblePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.argparser import get_argparser

parser = get_argparser()
parser.add_argument("--num_teachers", type=int, default=8)

if __name__ == "__main__":
    # For multiprocessing and float32 support, recommended to include at top of script
    freeze_support()
    to.set_default_dtype(to.float32)

    # Parse command line arguments
    args = parser.parse_args()

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True) 
    use_cuda = args.device == "cuda"
 
    # Environment
    env_hparams = dict(dt=1 / 250.0, max_steps=4000)
    env = ActNormWrapper(QQubeSwingUpSim(**env_hparams))

    # Policy
    policy_hparam = dict(num_policies=args.num_teachers, use_cuda=use_cuda)
    policy = EnsemblePolicy(spec=env.spec, **policy_hparam)

    # Test policy
    while True:
        input('Press some key to continue:')
        ro = rollout(
            env,
            policy,
            render_mode=RenderMode(text=True, video=True)
        )

        print(f'Return: {ro.undiscounted_return()}')
