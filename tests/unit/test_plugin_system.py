"""
Tests for the NEXUS Plugin System — manifest, loader, and sandbox.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
import yaml

from nexus.plugins.manifest import (
    PluginManifest,
    PluginType,
    load_manifest,
    validate_manifest,
)
from nexus.plugins.loader import PluginLoader
from nexus.plugins.sandbox import SandboxedExecutor


# ── Manifest Tests ───────────────────────────────────────────────────────────

class TestPluginManifest:

    def test_load_manifest(self, tmp_path):
        """Load a valid manifest from YAML file."""
        manifest_data = {
            "name": "test-plugin",
            "version": "1.0.0",
            "type": "agent",
            "description": "A test plugin",
            "entry_point": "nexus.plugins.test:TestAgent",
            "dependencies": ["httpx"],
            "config": {"timeout": 30},
            "enabled": True,
        }
        manifest_file = tmp_path / "manifest.yaml"
        manifest_file.write_text(yaml.dump(manifest_data))

        manifest = load_manifest(manifest_file)

        assert manifest.name == "test-plugin"
        assert manifest.version == "1.0.0"
        assert manifest.type == PluginType.AGENT
        assert manifest.description == "A test plugin"
        assert manifest.entry_point == "nexus.plugins.test:TestAgent"
        assert manifest.dependencies == ["httpx"]
        assert manifest.config == {"timeout": 30}
        assert manifest.enabled is True

    def test_load_manifest_missing_required_field(self, tmp_path):
        """Should raise ValueError for missing required fields."""
        manifest_data = {"name": "broken", "version": "1.0.0"}
        manifest_file = tmp_path / "manifest.yaml"
        manifest_file.write_text(yaml.dump(manifest_data))

        with pytest.raises(ValueError, match="missing required field"):
            load_manifest(manifest_file)

    def test_load_manifest_invalid_type(self, tmp_path):
        """Should raise ValueError for invalid plugin type."""
        manifest_data = {
            "name": "broken",
            "version": "1.0.0",
            "type": "invalid_type",
        }
        manifest_file = tmp_path / "manifest.yaml"
        manifest_file.write_text(yaml.dump(manifest_data))

        with pytest.raises(ValueError, match="Invalid plugin type"):
            load_manifest(manifest_file)

    def test_load_manifest_not_found(self):
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_manifest("/nonexistent/manifest.yaml")

    def test_validate_manifest_valid(self):
        """Valid manifest should have no errors."""
        manifest = PluginManifest(
            name="test",
            version="1.0.0",
            type=PluginType.SKILL,
            entry_point="module:func",
        )
        errors = validate_manifest(manifest)
        assert errors == []

    def test_validate_manifest_empty_name(self):
        """Empty name should produce an error."""
        manifest = PluginManifest(name="", version="1.0.0", type=PluginType.AGENT)
        errors = validate_manifest(manifest)
        assert any("name" in e.lower() for e in errors)

    def test_validate_manifest_empty_version(self):
        """Empty version should produce an error."""
        manifest = PluginManifest(name="test", version="", type=PluginType.AGENT)
        errors = validate_manifest(manifest)
        assert any("version" in e.lower() for e in errors)

    def test_validate_manifest_bad_entry_point(self):
        """Entry point without colon should produce an error."""
        manifest = PluginManifest(
            name="test", version="1.0.0", type=PluginType.AGENT,
            entry_point="no_colon_here",
        )
        errors = validate_manifest(manifest)
        assert any("entry_point" in e for e in errors)


# ── Loader Tests ─────────────────────────────────────────────────────────────

class TestPluginLoader:

    def test_plugin_loader_discover(self, tmp_path):
        """Discover plugins from directories containing manifest.yaml."""
        # Create a plugin directory with a manifest
        plugin_dir = tmp_path / "agents"
        plugin_dir.mkdir()
        plugin_subdir = plugin_dir / "my_agent"
        plugin_subdir.mkdir()
        manifest_data = {
            "name": "my-agent",
            "version": "0.1.0",
            "type": "agent",
            "description": "Test agent plugin",
        }
        (plugin_subdir / "manifest.yaml").write_text(yaml.dump(manifest_data))

        loader = PluginLoader(plugin_dirs=[str(plugin_dir)])
        manifests = loader.discover()

        assert len(manifests) == 1
        assert manifests[0].name == "my-agent"
        assert manifests[0].type == PluginType.AGENT

    def test_plugin_loader_load_all_empty(self, tmp_path):
        """Discovering from empty dirs should return no plugins."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        loader = PluginLoader(plugin_dirs=[str(empty_dir)])
        results = loader.load_all()

        assert results == {}

    def test_plugin_loader_get_loaded(self, tmp_path):
        """After loading, get_loaded should return manifests."""
        loader = PluginLoader(plugin_dirs=[])
        manifest = PluginManifest(
            name="test-plugin",
            version="1.0.0",
            type=PluginType.SKILL,
        )
        loader.load(manifest)

        loaded = loader.get_loaded()
        assert "test-plugin" in loaded
        assert loaded["test-plugin"].version == "1.0.0"

    def test_plugin_loader_unload(self):
        """Unloading a loaded plugin should remove it."""
        loader = PluginLoader(plugin_dirs=[])
        manifest = PluginManifest(
            name="to-unload",
            version="1.0.0",
            type=PluginType.GUARDRAIL,
        )
        loader.load(manifest)
        assert "to-unload" in loader.get_loaded()

        result = loader.unload("to-unload")
        assert result is True
        assert "to-unload" not in loader.get_loaded()

    def test_plugin_loader_unload_not_found(self):
        """Unloading a non-existent plugin should return False."""
        loader = PluginLoader(plugin_dirs=[])
        result = loader.unload("does-not-exist")
        assert result is False


# ── Sandbox Tests ────────────────────────────────────────────────────────────

class TestSandboxedExecutor:

    @pytest.mark.asyncio
    async def test_sandboxed_executor_success(self):
        """Normal function execution should return the result."""
        executor = SandboxedExecutor(timeout=5.0)

        def add(a, b):
            return a + b

        result = await executor.execute(add, 2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_sandboxed_executor_async_success(self):
        """Async function execution should return the result."""
        executor = SandboxedExecutor(timeout=5.0)

        async def async_add(a, b):
            return a + b

        result = await executor.execute(async_add, 10, 20)
        assert result == 30

    @pytest.mark.asyncio
    async def test_sandboxed_executor_timeout(self):
        """Long-running function should raise TimeoutError."""
        executor = SandboxedExecutor(timeout=0.1)

        async def slow_func():
            await asyncio.sleep(10)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            await executor.execute(slow_func)

    @pytest.mark.asyncio
    async def test_sandboxed_executor_exception(self):
        """Exceptions from the function should propagate."""
        executor = SandboxedExecutor(timeout=5.0)

        def failing():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await executor.execute(failing)
