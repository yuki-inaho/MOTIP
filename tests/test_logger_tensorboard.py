"""Tests for the Logger's optional TensorBoard layer (porting item 2).

Contract:
  - Default (TENSORBOARD off): writer is None, no <logdir>/tb directory is
    created, and log.txt behaviour is unchanged (non-destructive).
  - When enabled on the main process: a SummaryWriter is created under
    <logdir>/tb and add_scalar calls land in event files.
"""

from pathlib import Path

from log.logger import Logger


def test_writer_disabled_by_default(tmp_path: Path) -> None:
    logger = Logger(logdir=str(tmp_path / "run"), use_wandb=False)
    assert logger.tb_writer is None
    # No tb directory must be created when disabled.
    assert not (tmp_path / "run" / "tb").exists()


def test_writer_enabled_creates_tb_dir(tmp_path: Path) -> None:
    logger = Logger(logdir=str(tmp_path / "run"), use_wandb=False, tensorboard=True)
    assert logger.tb_writer is not None
    assert (tmp_path / "run" / "tb").is_dir()
    logger.close()


def test_tb_scalar_writes_event_file(tmp_path: Path) -> None:
    logger = Logger(logdir=str(tmp_path / "run"), use_wandb=False, tensorboard=True)
    logger.tb_scalar("Loss/total", 1.23, global_step=0)
    logger.tb_scalar("Loss/total", 0.99, global_step=1)
    logger.close()
    # At least one tfevents file exists.
    events = list((tmp_path / "run" / "tb").glob("*tfevents*"))
    assert len(events) >= 1


def test_tb_scalar_is_noop_when_disabled(tmp_path: Path) -> None:
    logger = Logger(logdir=str(tmp_path / "run"), use_wandb=False)
    # Must not raise and must not create any tb artifacts.
    logger.tb_scalar("Loss/total", 1.0, global_step=0)
    assert not (tmp_path / "run" / "tb").exists()
