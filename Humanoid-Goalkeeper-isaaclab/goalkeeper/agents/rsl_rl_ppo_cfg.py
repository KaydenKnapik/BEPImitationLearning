"""RSL-RL OnPolicyRunner configuration for the Goalkeeper environment.

Uses Isaac Lab's rsl_rl wrapper with standard PPO hyperparameters
matching the original G1 HIM-PPO configuration.

Compatible with rsl_rl >= 5.0.0 (uses actor/critic + distribution_cfg API).
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlMLPModelCfg,
)


@configclass
class GoalkeeperPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Standard PPO runner config — mirrors g1_29 HIM-PPO hyperparameters."""

    num_steps_per_env: int = 100
    max_iterations: int = 200000
    save_interval: int = 200
    experiment_name: str = "goalkeeper"
    run_name: str = "g1_isaaclab"
    logger: str = "tensorboard"
    wandb_project: str = "goalkeeper"

    # rsl_rl 5.x: map env observation dict keys to algorithm observation sets
    # "actor" receives the rolling history obs (960-dim), "critic" the privileged obs (113-dim)
    obs_groups: dict = {
        "actor": ["policy"],
        "critic": ["critic"],
    }

    # Required by RslRlBaseRunnerCfg in rsl_rl 5.x
    empirical_normalization: bool = False

    clip_actions: float = 100.0

    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    # Actor: stochastic MLP with Gaussian output
    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 256],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )

    # Critic: deterministic MLP
    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 256],
        activation="elu",
        obs_normalization=False,
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
