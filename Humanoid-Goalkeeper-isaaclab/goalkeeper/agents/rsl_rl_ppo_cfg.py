"""RSL-RL OnPolicyRunner configuration for the Goalkeeper environment."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
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

    # Map env observation dict keys to rsl_rl algorithm keys
    obs_groups: dict = {
        "policy": ["policy"],   # actor obs (960-dim history)
        "critic": ["critic"],   # critic obs (113-dim privileged)
    }

    clip_actions: float = 100.0

    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 256],
        critic_hidden_dims=[512, 256, 256],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        noise_std_type="scalar",
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
