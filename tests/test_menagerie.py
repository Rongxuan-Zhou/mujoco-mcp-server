import pytest
import json
from mujoco_mcp.menagerie_loader import MenagerieLoader

def test_available_categories():
    loader = MenagerieLoader()
    models = loader.get_available_models()
    assert "arms" in models
    assert "quadrupeds" in models
    assert "franka_emika_panda" in models["arms"]

def test_list_by_category():
    loader = MenagerieLoader()
    arms = loader.get_available_models()["arms"]
    assert len(arms) > 0

def test_loader_cache_dir_created(tmp_path):
    loader = MenagerieLoader(cache_dir=str(tmp_path / "cache"))
    assert (tmp_path / "cache").exists()

@pytest.mark.network
def test_validate_model_franka():
    loader = MenagerieLoader()
    result = loader.validate_model("franka_emika_panda")
    assert result["valid"] is True
    assert result["n_bodies"] > 0
