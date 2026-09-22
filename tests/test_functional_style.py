import ast
import asyncio
import unittest
from pathlib import Path

from IBMRuntime import scan_left
from IBMRuntime.effects.async_ import wait_until
from IBMRuntime.result import Ok


class FunctionalStyleTests(unittest.TestCase):
    def test_scan_left_returns_every_immutable_state(self):
        self.assertEqual(
            scan_left(lambda total, value: total + value, 0, (1, 2, 3)),
            (0, 1, 3, 6),
        )

    def test_ibm_runtime_has_no_imperative_iteration_syntax(self):
        prohibited = (
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.AugAssign,
        )
        package = Path(__file__).parents[1] / "IBMRuntime"
        violations = tuple(
            (source.name, type(node).__name__, node.lineno)
            for source in package.rglob("*.py")
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
            if isinstance(node, prohibited)
        )

        self.assertEqual(violations, ())


class AsyncTraversalTests(unittest.IsolatedAsyncioTestCase):
    async def test_polling_trampoline_handles_multiple_pending_states(self):
        statuses = iter((Ok(False), Ok(False), Ok(True)))
        result = await asyncio.wait_for(
            wait_until(
                lambda: next(statuses),
                lambda: Ok("complete"),
                poll_interval=0.0,
                poll_in_thread=False,
            ),
            timeout=1.0,
        )

        self.assertEqual(result, Ok("complete"))


if __name__ == "__main__":
    unittest.main()
