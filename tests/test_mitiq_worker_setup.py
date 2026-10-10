"""Fast stdlib-only preparation tests: never start/import the Mitiq worker."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_file("oqs_worker_setup_test", ROOT / "mitiq_qem/setup_worker.py")
environment = load_file("oqs_worker_environment_test", ROOT / "mitiq_qem/_environment.py")


class PreparationTests(unittest.TestCase):
    def test_default_existing_python_is_worker_not_main(self):
        with patch.object(setup, "compile_existing") as compile_existing:
            setup.main(["--compile-existing"])
        self.assertEqual(compile_existing.call_args.args[0], setup.python_in(ROOT / ".venv-mitiq"))

    def test_compile_dry_run_never_starts_subprocess_or_inspects_interpreter(self):
        with patch.object(setup, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            setup.compile_existing(Path("missing/python"), dry_run=True)
        run.assert_not_called()

    def test_compilation_is_incremental_and_uses_selected_python(self):
        python = Path("worker/bin/python")
        paths = (Path("worker/lib/site-packages"),)
        with patch.object(setup, "inspect_python", return_value=({}, paths)), \
                patch.object(setup, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            setup.compile_existing(python)
        command = run.call_args.args[0]
        self.assertEqual(command[:4], (python, "-m", "compileall", "-q"))
        self.assertNotIn("-f", command)
        self.assertIn(ROOT / "mitiq_qem", command)

    def test_local_paths_are_explicit_and_both_installs_compile_bytecode(self):
        directory = (Path(tempfile.gettempdir()) / "oqs-worker-preparation-test").absolute()
        with patch.dict(os.environ, {"VIRTUAL_ENV": "main", "UV_PYTHON": "main"}):
            python, env, commands = setup.local_plan(directory)
        self.assertEqual(python, setup.python_in(directory / "venv"))
        self.assertEqual(env["UV_PYTHON_INSTALL_DIR"], str(directory / "python"))
        self.assertEqual(env["UV_CACHE_DIR"], str(directory / "uv-cache"))
        self.assertNotIn("VIRTUAL_ENV", env)
        self.assertNotIn("UV_PYTHON", env)
        self.assertIn("--compile-bytecode", commands[0])
        self.assertIn("--compile-bytecode", commands[2])
        self.assertIn("--no-bin", commands[0])
        self.assertIn("--no-registry", commands[0])
        self.assertNotIn("--clear", commands[1])
        self.assertIn(setup.REQUIREMENT, commands[2])

    def test_offline_and_frozen_requirements_are_forwarded(self):
        _, _, commands = setup.local_plan("/tmp/worker", offline=True, requirements="deps.txt")
        self.assertTrue(all("--offline" in command for command in commands))
        self.assertIn("--requirements", commands[2])
        self.assertIn(Path("deps.txt").absolute(), commands[2])

    def test_local_dry_run_creates_nothing_and_runs_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "new-worker"
            with patch.object(setup, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                setup.prepare_local(directory, dry_run=True)
            self.assertFalse(directory.exists())
            run.assert_not_called()

    def test_unrelated_nonempty_directory_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sentinel = directory / "keep.txt"
            sentinel.write_text("keep")
            with patch.object(setup.shutil, "which", return_value="uv"), \
                    patch.object(setup, "run") as run:
                with self.assertRaisesRegex(ValueError, "empty local directory"):
                    setup.prepare_local(directory)
            self.assertEqual(sentinel.read_text(), "keep")
            self.assertFalse((directory / setup.MARKER).exists())
            run.assert_not_called()

    def test_system_python_is_rejected(self):
        python = Path(sys.executable)
        details = {"prefix": "/python", "base_prefix": "/python", "version": [3, 12],
                   "paths": ["/python/site-packages", "/python/site-packages"]}
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps(details))
        with patch.object(setup, "run", return_value=result):
            with self.assertRaisesRegex(ValueError, "system Python"):
                setup.inspect_python(python)

    def test_main_environment_is_rejected(self):
        details = {"prefix": str(ROOT / ".venv"), "base_prefix": "/python", "version": [3, 12],
                   "paths": [str(ROOT / ".venv/lib/site-packages")]}
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps(details))
        with patch.object(setup, "run", return_value=result):
            with self.assertRaisesRegex(ValueError, "must remain unchanged"):
                setup.inspect_python(Path(sys.executable))

    def test_supported_worker_paths_are_deduplicated(self):
        prefix = (Path(tempfile.gettempdir()) / "worker").resolve()
        details = {"prefix": str(prefix), "base_prefix": "/python", "version": [3, 12],
                   "paths": [str(prefix / "site-packages")] * 2}
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps(details))
        with patch.object(setup, "run", return_value=result):
            _, paths = setup.inspect_python(Path(sys.executable))
        self.assertEqual(paths, (prefix / "site-packages",))

    def test_local_base_python_is_required_before_dependency_install(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(setup.shutil, "which", return_value="uv"), \
                patch.object(setup, "run") as run, \
                patch.object(setup, "inspect_python", return_value=({"base_prefix": "/shared/python"}, ())):
            with self.assertRaisesRegex(ValueError, "base Python is not local"):
                setup.prepare_local(Path(temporary) / "worker")
            self.assertEqual(run.call_count, 2)

    def test_module_loading_does_not_import_scientific_packages(self):
        before = set(sys.modules)
        load_file("oqs_worker_setup_reimport_test", ROOT / "mitiq_qem/setup_worker.py")
        load_file("oqs_worker_environment_reimport_test", ROOT / "mitiq_qem/_environment.py")
        added = set(sys.modules) - before
        self.assertFalse(added & {"mitiq", "cirq", "sympy"})


class EnvironmentTests(unittest.TestCase):
    def test_worker_passes_cache_environment_to_child_and_reports_it(self):
        # Load worker.py with a tiny wire stub; never import Qiskit/Mitiq.
        package = types.ModuleType("oqs_worker_test_package")
        package.__path__ = [str(ROOT / "mitiq_qem")]
        wire = types.ModuleType("oqs_worker_test_package._wire")
        wire.PROTOCOL_VERSION = 1
        wire.decode = wire.encode = lambda value: value
        wire.qpy_range = lambda: (1, 2)
        modules = {package.__name__: package, wire.__name__: wire,
                   package.__name__ + "._environment": environment}
        with patch.dict(sys.modules, modules), patch("atexit.register"):
            worker_module = load_file(package.__name__ + ".worker", ROOT / "mitiq_qem/worker.py")
        from unittest.mock import MagicMock
        child = MagicMock()
        child.stdin = io.StringIO()
        child.stdout = io.StringIO(json.dumps({"protocol": 1, "qpy_min": 1, "qpy_max": 2}) + "\n")
        child.poll.return_value = 0
        prepared = {"MPLCONFIGDIR": "/prepared/cache"}
        with patch.object(worker_module, "worker_environment", return_value=prepared), \
                patch.object(worker_module.subprocess, "Popen", return_value=child) as popen, \
                contextlib.redirect_stderr(io.StringIO()):
            worker = worker_module._Worker(Path("worker/python"), 5, 5)
            try:
                self.assertEqual(popen.call_args.kwargs["env"], prepared)
                self.assertEqual(worker.info["matplotlib_cache"], "/prepared/cache")
            finally:
                worker.close()

    def test_user_matplotlib_setting_is_preserved_without_cache_setup(self):
        with patch.dict(os.environ, {"MPLCONFIGDIR": "/chosen/cache"}), \
                patch.object(environment.tempfile, "gettempdir") as gettempdir:
            result = environment.worker_environment(Path("worker/python"))
        self.assertEqual(result["MPLCONFIGDIR"], "/chosen/cache")
        gettempdir.assert_not_called()

    def test_cache_is_reused_and_parent_environment_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(os.environ, {}, clear=True), \
                patch.object(environment.tempfile, "gettempdir", return_value=temporary):
            first = environment.worker_environment(Path("worker/python"))
            second = environment.worker_environment(Path("worker/python"))
            other = environment.worker_environment(Path("other/python"))
            self.assertNotIn("MPLCONFIGDIR", os.environ)
            self.assertEqual(first["MPLCONFIGDIR"], second["MPLCONFIGDIR"])
            self.assertNotEqual(first["MPLCONFIGDIR"], other["MPLCONFIGDIR"])
            self.assertTrue(Path(first["MPLCONFIGDIR"]).is_dir())

    @unittest.skipUnless(os.name == "posix", "POSIX ownership protection")
    def test_world_writable_cache_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(os.environ, {}, clear=True), \
                patch.object(environment.tempfile, "gettempdir", return_value=temporary):
            value = environment.worker_environment(Path("worker/python"))
            Path(value["MPLCONFIGDIR"]).parent.chmod(0o777)
            with self.assertRaisesRegex(OSError, "private and owned"):
                environment.worker_environment(Path("worker/python"))

    @unittest.skipUnless(os.name == "posix", "POSIX symlink protection")
    def test_symlinked_matplotlib_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(os.environ, {}, clear=True), \
                patch.object(environment.tempfile, "gettempdir", return_value=temporary):
            value = environment.worker_environment(Path("worker/python"))
            cache = Path(value["MPLCONFIGDIR"])
            cache.rmdir()
            elsewhere = Path(temporary) / "elsewhere"
            elsewhere.mkdir()
            cache.symlink_to(elsewhere, target_is_directory=True)
            with self.assertRaisesRegex(OSError, "symlink"):
                environment.worker_environment(Path("worker/python"))


if __name__ == "__main__":
    unittest.main()
