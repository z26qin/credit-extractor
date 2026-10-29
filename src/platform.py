"""Platform detection, paths, secrets, and Hugging Face Hub sync.

Every notebook calls `setup()` first. It works unchanged on Kaggle, Colab, and a local
machine (CPU tests). The Hub is the single artifact store across platforms: a PRIVATE
dataset repo for data / chunk outputs / reports, and a PRIVATE model repo for adapters.

Imported as `src.platform` (it does not shadow the stdlib `platform` module because
`src/` itself is never put on sys.path, only its parent).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

SECRET_NAMES = ("HF_TOKEN", "SEC_USER_AGENT", "NEBIUS_API_KEY")

# Default Hub repo names; the owner is the HF_TOKEN account unless overridden.
DATA_REPO_NAME = "credit-extractor-data"
MODEL_REPO_NAME = "credit-extractor-adapters"


def detect_platform(environ: dict | None = None) -> str:
    """Return 'kaggle', 'colab', or 'local'."""
    env = os.environ if environ is None else environ
    if "KAGGLE_KERNEL_RUN_TYPE" in env or "KAGGLE_URL_BASE" in env:
        return "kaggle"
    if "COLAB_RELEASE_TAG" in env or "COLAB_GPU" in env:
        return "colab"
    return "local"


@dataclass
class Paths:
    work: Path   # persistent-for-the-session outputs (mirrored to the Hub)
    cache: Path  # large downloads (model weights); never uploaded

    def ensure(self) -> "Paths":
        self.work.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        return self


def platform_paths(platform: str, local_root: Path | None = None) -> Paths:
    if platform == "kaggle":
        # /kaggle/working is saved as notebook output (20 GB cap): keep weights out of it
        return Paths(work=Path("/kaggle/working/work"), cache=Path("/root/.cache/huggingface"))
    if platform == "colab":
        return Paths(work=Path("/content/work"), cache=Path("/root/.cache/huggingface"))
    root = local_root or Path.cwd()
    return Paths(work=root / "work", cache=Path.home() / ".cache" / "huggingface")


def get_secret(name: str, platform: str | None = None) -> str | None:
    """Read a secret from the platform's secret store, falling back to os.environ."""
    platform = platform or detect_platform()
    if os.environ.get(name):
        return os.environ[name]
    try:
        if platform == "kaggle":
            from kaggle_secrets import UserSecretsClient  # type: ignore
            return UserSecretsClient().get_secret(name)
        if platform == "colab":
            from google.colab import userdata  # type: ignore
            return userdata.get(name)
    except Exception:  # secret not attached / not granted to this notebook
        return None
    return None


def load_secrets(names=SECRET_NAMES, platform: str | None = None) -> dict[str, bool]:
    """Export available secrets to os.environ. Returns {name: found} (never the values)."""
    found = {}
    for n in names:
        v = get_secret(n, platform)
        if v:
            os.environ[n] = v
        found[n] = bool(v)
    return found


def gpu_info() -> list[dict]:
    """[{name, memory_mib, compute_capability}] via nvidia-smi; [] when no GPU."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "memory_mib": int(float(parts[1])),
                         "compute_capability": parts[2]})
    return gpus


@dataclass
class Env:
    platform: str
    paths: Paths
    secrets: dict[str, bool]
    gpus: list[dict]
    data_repo: str | None = None
    model_repo: str | None = None
    started_at: float = field(default_factory=time.time)

    def session_minutes(self) -> float:
        return (time.time() - self.started_at) / 60


def setup(local_root: Path | None = None, hub: bool = True,
          data_repo: str | None = None, model_repo: str | None = None) -> Env:
    """Detect platform, set paths and secrets, and (optionally) ensure the private Hub repos."""
    plat = detect_platform()
    paths = platform_paths(plat, local_root).ensure()
    os.environ.setdefault("HF_HOME", str(paths.cache))
    secrets = load_secrets(platform=plat)
    env = Env(platform=plat, paths=paths, secrets=secrets, gpus=gpu_info())
    if hub and secrets.get("HF_TOKEN"):
        env.data_repo, env.model_repo = ensure_hub_repos(data_repo, model_repo)
    print(f"platform={plat} work={paths.work} gpus={[g['name'] for g in env.gpus]} "
          f"secrets={secrets} data_repo={env.data_repo} model_repo={env.model_repo}")
    if plat == "kaggle" and env.gpus and any(g["compute_capability"] < "7.0" for g in env.gpus):
        print("WARNING: a GPU with compute capability < 7.0 (P100?) - vLLM will not run. "
              "Switch the accelerator to 'GPU T4 x2'.")
    return env


# ---------------------------------------------------------------- Hub sync helpers

def _api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def ensure_hub_repos(data_repo: str | None = None,
                     model_repo: str | None = None) -> tuple[str, str]:
    """Create (if missing) the PRIVATE dataset + model repos. Returns their ids."""
    api = _api()
    owner = api.whoami()["name"]
    data_repo = data_repo or f"{owner}/{DATA_REPO_NAME}"
    model_repo = model_repo or f"{owner}/{MODEL_REPO_NAME}"
    api.create_repo(data_repo, repo_type="dataset", private=True, exist_ok=True)
    api.create_repo(model_repo, repo_type="model", private=True, exist_ok=True)
    for rid, rtype in ((data_repo, "dataset"), (model_repo, "model")):
        if not api.repo_info(rid, repo_type=rtype).private:
            raise RuntimeError(f"{rtype} repo {rid} exists but is PUBLIC - refusing to use it")
    return data_repo, model_repo


def pull(repo_id: str, prefix: str, local_dir: Path, repo_type: str = "dataset") -> int:
    """Download everything under `prefix/` into local_dir/prefix. Returns #files pulled.

    Used at notebook start for resumability: finished chunk outputs are skipped later.
    """
    from huggingface_hub import snapshot_download
    files = [f for f in _api().list_repo_files(repo_id, repo_type=repo_type)
             if f.startswith(prefix.rstrip("/") + "/")]
    if not files:
        return 0
    snapshot_download(repo_id, repo_type=repo_type, local_dir=str(local_dir),
                      allow_patterns=[prefix.rstrip("/") + "/**"],
                      token=os.environ.get("HF_TOKEN"))
    return len(files)


def push_folder(repo_id: str, local_dir: Path, path_in_repo: str,
                message: str = "sync", repo_type: str = "dataset") -> None:
    _api().upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(local_dir),
                         path_in_repo=path_in_repo, commit_message=message)


def push_file(repo_id: str, local_path: Path, path_in_repo: str,
              message: str = "upload", repo_type: str = "dataset") -> None:
    _api().upload_file(repo_id=repo_id, repo_type=repo_type, path_or_fileobj=str(local_path),
                       path_in_repo=path_in_repo, commit_message=message)


def start_sync(repo_id: str, local_dir: Path, path_in_repo: str,
               every_minutes: float = 3, repo_type: str = "dataset"):
    """Background push of local_dir every N minutes (only changed files are uploaded).

    A disconnect therefore loses at most ~N minutes of outputs. Call
    `scheduler.trigger().result()` at the end of a stage to force a final push.
    """
    from huggingface_hub import CommitScheduler
    local_dir.mkdir(parents=True, exist_ok=True)
    return CommitScheduler(repo_id=repo_id, repo_type=repo_type, folder_path=str(local_dir),
                           path_in_repo=path_in_repo, every=every_minutes, private=True,
                           ignore_patterns=["*.tmp"], token=os.environ.get("HF_TOKEN"))
