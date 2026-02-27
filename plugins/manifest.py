"""
NEXUS Plugin Manifest — Plugin Metadata & Validation
Defines the PluginManifest dataclass and utilities for loading/validating
plugin manifest YAML files.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("nexus.plugins.manifest")


# ── Plugin Types ─────────────────────────────────────────────────────────────

class PluginType(str, Enum):
    """Supported plugin types in the NEXUS ecosystem."""
    AGENT = "agent"
    SKILL = "skill"
    MCP_SERVER = "mcp_server"
    GUARDRAIL = "guardrail"


# ── Plugin Manifest ──────────────────────────────────────────────────────────

@dataclass
class PluginManifest:
    """
    Metadata describing a NEXUS plugin.
    Loaded from a manifest.yaml file in each plugin directory.
    """
    name: str
    version: str
    type: PluginType
    description: str = ""
    entry_point: str = ""           # module:class or module:function
    dependencies: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = PluginType(self.type)


# ── Manifest Loader ──────────────────────────────────────────────────────────

def load_manifest(path: str | Path) -> PluginManifest:
    """
    Parse a YAML manifest file and return a PluginManifest.

    Args:
        path: Path to a manifest.yaml file.

    Returns:
        PluginManifest instance.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If the manifest is missing required fields.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a YAML mapping, got {type(data).__name__}")

    # Required fields
    for required in ("name", "version", "type"):
        if required not in data:
            raise ValueError(f"Manifest missing required field: '{required}' in {path}")

    # Convert type string to enum
    try:
        plugin_type = PluginType(data["type"])
    except ValueError:
        valid = ", ".join(t.value for t in PluginType)
        raise ValueError(f"Invalid plugin type '{data['type']}'; must be one of: {valid}")

    return PluginManifest(
        name=data["name"],
        version=data["version"],
        type=plugin_type,
        description=data.get("description", ""),
        entry_point=data.get("entry_point", ""),
        dependencies=data.get("dependencies", []),
        config=data.get("config", {}),
        enabled=data.get("enabled", True),
    )


# ── Manifest Validator ───────────────────────────────────────────────────────

def validate_manifest(manifest: PluginManifest) -> list[str]:
    """
    Validate a PluginManifest and return a list of error messages.
    An empty list means the manifest is valid.
    """
    errors: list[str] = []

    if not manifest.name or not manifest.name.strip():
        errors.append("Plugin name must not be empty")

    if not manifest.version or not manifest.version.strip():
        errors.append("Plugin version must not be empty")

    if not isinstance(manifest.type, PluginType):
        errors.append(f"Invalid plugin type: {manifest.type}")

    if manifest.entry_point:
        # Validate entry_point format: module:class or module:function
        if ":" not in manifest.entry_point:
            errors.append(
                f"entry_point must be in 'module:class' format, got '{manifest.entry_point}'"
            )

    if not isinstance(manifest.dependencies, list):
        errors.append("dependencies must be a list")

    if not isinstance(manifest.config, dict):
        errors.append("config must be a dict")

    return errors
