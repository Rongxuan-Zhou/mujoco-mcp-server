import pytest

def test_rl_env_creates_spaces():
    pytest.importorskip("gymnasium")
    from mujoco_mcp.rl_integration import MuJoCoRLEnvironment, RLConfig, TaskType
    config = RLConfig(
        robot_type="franka_panda",
        task_type=TaskType.REACHING,
        max_episode_steps=100,
        observation_space_size=14,
        action_space_size=7,
    )
    env = MuJoCoRLEnvironment(config)
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.action_space.shape == (7,)

def test_rl_config_defaults():
    pytest.importorskip("gymnasium")
    from mujoco_mcp.rl_integration import RLConfig, TaskType
    config = RLConfig(
        robot_type="franka_panda",
        task_type=TaskType.REACHING,
        max_episode_steps=500,
        observation_space_size=14,
        action_space_size=7,
    )
    assert config.robot_type == "franka_panda"
    assert config.max_episode_steps == 500
