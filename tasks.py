import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "tero_project"
PYTHON_VERSION = "3.10"

# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw/corruptmnist_v1 data/processed", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run python -m src.{PROJECT_NAME}.train", echo=True, pty=not WINDOWS)

@task
def eval(ctx: Context, model_checkpoint: str = "models/model.pth") -> None:
    """Evaluate the model."""
    ctx.run(f"uv run python -c \"from src.{PROJECT_NAME}.evaluate import evaluate; evaluate('{model_checkpoint}')\"", echo=True, pty=not WINDOWS)


@task
def visualize(ctx: Context, model_checkpoint: str = "models/model.pth", figure_name: str = "embeddings.png") -> None:
    """Visualize model embeddings."""
    ctx.run(f"uv run python -c \"from src.{PROJECT_NAME}.visualize import visualize; visualize('{model_checkpoint}', '{figure_name}')\"", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task
def git(ctx: Context, message: str) -> None:
    """Add, commit, and push changes."""
    ctx.run("git add .", echo=True, pty=not WINDOWS)
    ctx.run(f'git commit -m "{message}"', echo=True, pty=not WINDOWS)
    ctx.run("git push", echo=True, pty=not WINDOWS)



