import tempfile

import pytest
import yaml

from src.config import get_config, reload_config, set_config_path


class TestConfigDefaults:
    def test_get_config_returns_dict(self):
        reload_config()
        cfg = get_config()
        assert isinstance(cfg, dict)

    def test_default_sections_exist(self):
        reload_config()
        cfg = get_config()
        assert "analysis" in cfg
        assert "explanation" in cfg
        assert "rate_limiting" in cfg
        assert "cache" in cfg
        assert "model" in cfg

    def test_analysis_defaults(self):
        reload_config()
        cfg = get_config()
        assert cfg["analysis"]["top_k_neurons"] == 20
        assert cfg["analysis"]["top_k_heads"] == 10
        assert cfg["analysis"]["context_window_size"] == 5

    def test_explanation_defaults(self):
        reload_config()
        cfg = get_config()
        assert cfg["explanation"]["provider"] == "groq"
        assert cfg["explanation"]["batch_size"] == 5


class TestConfigOverride:
    def test_config_overrides_from_file(self):
        overrides = {
            "analysis": {"top_k_neurons": 5},
            "explanation": {"provider": "claude"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(overrides, f)
            f.flush()
            config_path = f.name

        try:
            set_config_path(config_path)
            reload_config()
            cfg = get_config()
            assert cfg["analysis"]["top_k_neurons"] == 5
            assert cfg["explanation"]["provider"] == "claude"
            assert cfg["analysis"]["top_k_heads"] == 10
        finally:
            import os
            os.unlink(config_path)

    def test_partial_override_keeps_other_defaults(self):
        overrides = {
            "analysis": {"top_k_neurons": 50},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(overrides, f)
            f.flush()
            config_path = f.name

        try:
            set_config_path(config_path)
            reload_config()
            cfg = get_config()
            assert cfg["analysis"]["top_k_neurons"] == 50
            assert cfg["analysis"]["context_window_size"] == 5
        finally:
            import os
            os.unlink(config_path)
