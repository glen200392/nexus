"""
NEXUS Plugin Loader — Dynamic Plugin Discovery & Loading
Scans plugin directories for manifest.yaml files, validates them,
and dynamically imports plugin entry points.
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

from nexus.plugins.manifest import (
    PluginManifest,
    PluginType,
    load_manifest,
    validate_manifest,
)

logger = logging.getLogger("nexus.plugins.loader")

# Default plugin directories (relative to project root)
_DEFAULT_PLUGIN_DIRS = [
    "plugins/agents",
    "plugins/skills",
    "plugins/mcp",
    "plugins/guardrails",
]


class PluginLoader:
    """
    Discovers, loads, and manages NEXUS plugins.

    Plugins are discovered by scanning plugin directories for manifest.yaml files.
    Each plugin is dynamically imported via its entry_point field.
    """

    def __init__(self, plugin_dirs: list[str | Path] | None = None):
        """
        Initialize the plugin loader.

        Args:
            plugin_dirs: List of directories to scan for plugins.
                         Defaults to the standard plugin directories.
        """
        if plugin_dirs is not None:
            self._plugin_dirs = [Path(d) for d in plugin_dirs]
        else:
            self._plugin_dirs = [Path(d) for d in _DEFAULT_PLUGIN_DIRS]

        self._loaded: dict[str, dict[str, Any]] = {}  # name -> {"manifest": ..., "instance": ...}
        self._manifests: dict[str, PluginManifest] = {}

    # ── Discovery ────────────────────────────────────────────────────────────

    def discover(self) -> list[PluginManifest]:
        """
        Scan plugin directories for manifest.yaml files.

        Returns:
            List of discovered PluginManifest objects.
        """
        manifests: list[PluginManifest] = []

        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists():
                logger.debug("Plugin directory does not exist: %s", plugin_dir)
                continue

            # Look for manifest.yaml in immediate subdirectories
            for child in sorted(plugin_dir.iterdir()):
                if not child.is_dir():
                    continue
                manifest_path = child / "manifest.yaml"
                if not manifest_path.exists():
                    continue

                try:
                    manifest = load_manifest(manifest_path)
                    errors = validate_manifest(manifest)
                    if errors:
                        logger.warning(
                            "Plugin '%s' manifest has errors: %s",
                            child.name, "; ".join(errors),
                        )
                        continue
                    manifests.append(manifest)
                    logger.debug("Discovered plugin: %s v%s", manifest.name, manifest.version)
                except Exception as exc:
                    logger.warning("Failed to load manifest from %s: %s", manifest_path, exc)

            # Also check the directory itself (for flat plugin layout)
            manifest_path = plugin_dir / "manifest.yaml"
            if manifest_path.exists():
                try:
                    manifest = load_manifest(manifest_path)
                    errors = validate_manifest(manifest)
                    if not errors:
                        manifests.append(manifest)
                except Exception as exc:
                    logger.warning("Failed to load manifest from %s: %s", manifest_path, exc)

        return manifests

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self, manifest: PluginManifest) -> Any:
        """
        Dynamically import and instantiate a plugin from its manifest.

        Args:
            manifest: The plugin manifest describing the entry point.

        Returns:
            The loaded plugin instance (class instance or function).

        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the class/function is not found in the module.
            ValueError: If the entry_point format is invalid.
        """
        if not manifest.enabled:
            logger.info("Plugin '%s' is disabled, skipping", manifest.name)
            return None

        if not manifest.entry_point:
            logger.info("Plugin '%s' has no entry_point, registering manifest only", manifest.name)
            self._manifests[manifest.name] = manifest
            self._loaded[manifest.name] = {"manifest": manifest, "instance": None}
            return None

        # Parse entry_point: "module.path:ClassName"
        if ":" not in manifest.entry_point:
            raise ValueError(
                f"Invalid entry_point format '{manifest.entry_point}'; "
                f"expected 'module:class' or 'module:function'"
            )

        module_path, attr_name = manifest.entry_point.rsplit(":", 1)

        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            logger.error("Failed to import plugin module '%s': %s", module_path, exc)
            raise

        attr = getattr(module, attr_name, None)
        if attr is None:
            raise AttributeError(
                f"Module '{module_path}' has no attribute '{attr_name}'"
            )

        # Instantiate if it's a class, otherwise use as-is (function)
        if isinstance(attr, type):
            try:
                instance = attr(**manifest.config)
            except TypeError:
                instance = attr()
        else:
            instance = attr

        self._manifests[manifest.name] = manifest
        self._loaded[manifest.name] = {"manifest": manifest, "instance": instance}
        logger.info("Loaded plugin: %s v%s (%s)", manifest.name, manifest.version, manifest.type.value)
        return instance

    def load_all(self) -> dict[str, Any]:
        """
        Discover and load all enabled plugins.

        Returns:
            Dict mapping plugin name to loaded instance.
        """
        manifests = self.discover()
        results: dict[str, Any] = {}

        for manifest in manifests:
            if not manifest.enabled:
                continue
            try:
                instance = self.load(manifest)
                results[manifest.name] = instance
            except Exception as exc:
                logger.error("Failed to load plugin '%s': %s", manifest.name, exc)

        logger.info("Loaded %d plugins", len(results))
        return results

    # ── Unloading ────────────────────────────────────────────────────────────

    def unload(self, name: str) -> bool:
        """
        Remove a loaded plugin.

        Args:
            name: The plugin name to unload.

        Returns:
            True if the plugin was unloaded, False if not found.
        """
        if name not in self._loaded:
            logger.warning("Plugin '%s' not loaded, cannot unload", name)
            return False

        del self._loaded[name]
        self._manifests.pop(name, None)
        logger.info("Unloaded plugin: %s", name)
        return True

    # ── Query ────────────────────────────────────────────────────────────────

    def get_loaded(self) -> dict[str, PluginManifest]:
        """
        Return manifests of all currently loaded plugins.

        Returns:
            Dict mapping plugin name to PluginManifest.
        """
        return dict(self._manifests)

    # ── Reload ───────────────────────────────────────────────────────────────

    def reload(self, name: str) -> Any:
        """
        Unload and reload a plugin by name.

        Args:
            name: The plugin name to reload.

        Returns:
            The reloaded plugin instance.

        Raises:
            KeyError: If the plugin was not previously loaded.
        """
        if name not in self._loaded:
            raise KeyError(f"Plugin '{name}' is not loaded; cannot reload")

        manifest = self._loaded[name]["manifest"]

        # Invalidate the cached module so importlib re-imports it
        if manifest.entry_point and ":" in manifest.entry_point:
            module_path = manifest.entry_point.rsplit(":", 1)[0]
            if module_path in sys.modules:
                del sys.modules[module_path]

        self.unload(name)
        return self.load(manifest)
