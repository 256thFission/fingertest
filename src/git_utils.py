#!/usr/bin/env python3
"""
Git utilities for reproducibility tracking.
"""

import subprocess
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def get_git_info() -> Dict[str, any]:
    """Get comprehensive git information."""
    try:
        return {
            "hash": get_git_hash(),
            "hash_short": get_git_hash()[:7],
            "branch": get_git_branch(),
            "remote": get_git_remote(),
            "dirty": has_uncommitted_changes(),
            "untracked": has_untracked_files(),
            "message": get_last_commit_message(),
        }
    except Exception as e:
        logger.warning(f"Failed to get git info: {e}")
        return {"error": str(e)}


def get_git_hash() -> str:
    """Get current git commit hash."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL
    ).decode().strip()


def get_git_branch() -> str:
    """Get current git branch."""
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        stderr=subprocess.DEVNULL
    ).decode().strip()


def get_git_remote() -> Optional[str]:
    """Get git remote URL."""
    try:
        return subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return None


def get_last_commit_message() -> str:
    """Get last commit message."""
    return subprocess.check_output(
        ["git", "log", "-1", "--pretty=%B"],
        stderr=subprocess.DEVNULL
    ).decode().strip()


def has_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes."""
    result = subprocess.call(
        ["git", "diff-index", "--quiet", "HEAD"],
        stderr=subprocess.DEVNULL
    )
    return result != 0


def has_untracked_files() -> bool:
    """Check if there are untracked files."""
    result = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        stderr=subprocess.DEVNULL
    ).decode().strip()
    return len(result) > 0


def require_clean_git(strict: bool = True):
    """Require clean git state before experiment."""
    if has_uncommitted_changes():
        if strict:
            raise RuntimeError(
                "Git has uncommitted changes. "
                "Commit or stash before running experiment. "
                "Set git.require_clean=false to override."
            )
        else:
            logger.warning("⚠️  Git has uncommitted changes")

    if has_untracked_files():
        logger.warning("⚠️  Git has untracked files")
