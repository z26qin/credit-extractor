import os
from pathlib import Path

from src import platform as plat


def test_detect_platform():
    assert plat.detect_platform({"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}) == "kaggle"
    assert plat.detect_platform({"COLAB_RELEASE_TAG": "x"}) == "colab"
    assert plat.detect_platform({}) == "local"


def test_paths():
    assert plat.platform_paths("kaggle").work == Path("/kaggle/working/work")
    assert plat.platform_paths("colab").work == Path("/content/work")
    assert plat.platform_paths("local", Path("/tmp/x")).work == Path("/tmp/x/work")


def test_secret_env_fallback(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Jane Doe jane@example.com")
    assert plat.get_secret("SEC_USER_AGENT", "local") == "Jane Doe jane@example.com"
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    assert plat.get_secret("NEBIUS_API_KEY", "local") is None
    # kaggle/colab modules are absent locally: must degrade to None, not raise
    assert plat.get_secret("NEBIUS_API_KEY", "kaggle") is None
    assert plat.get_secret("NEBIUS_API_KEY", "colab") is None


def test_load_secrets_reports_presence_only(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    found = plat.load_secrets(platform="local")
    assert found == {"HF_TOKEN": True, "SEC_USER_AGENT": False, "NEBIUS_API_KEY": False}
    assert "hf_secret_value" not in repr(found)


def test_setup_local_without_hub(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    for k in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "COLAB_RELEASE_TAG", "COLAB_GPU"):
        monkeypatch.delenv(k, raising=False)
    env = plat.setup(local_root=tmp_path, hub=False)
    assert env.platform == "local" and env.paths.work.is_dir()
    assert env.data_repo is None
