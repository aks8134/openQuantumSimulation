"""Bounded QPY I/O, including indexed access to older monolithic archives.

The public qpy.load API builds a list for the complete archive; the newer Rust
decoder also reads the remaining file into a buffer. Use Qiskit's versioned
Python circuit decoder instead, with unbounded reads forbidden. This small
adapter to QPY internals is tested against sequential and offset-table formats.
The original archive is read-only; payload bytes are never rewritten.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import struct

from qiskit.qpy import binary_io, common, formats, type_keys


def file_sha256(path, chunk_size=1024*1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class _ReadGuard:
    def __init__(self, handle):
        self.handle = handle

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def read(self, size=-1):
        if size is None or size < 0:
            raise ValueError("QPY decoder attempted an unbounded archive read")
        return self.handle.read(size)


class Archive:
    def __init__(self, path, expected_count):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        with self.path.open("rb") as handle:
            first = handle.read(7)
            if len(first) != 7 or first[:6] != b"QISKIT":
                raise ValueError("not a complete QPY archive")
            self.version = first[6]
            if self.version > common.QPY_VERSION:
                raise ValueError("this QPY archive requires a newer Qiskit reader")
            self.pack = formats.FILE_HEADER_V10_PACK if self.version >= 10 else formats.FILE_HEADER_PACK
            header_type = formats.FILE_HEADER_V10 if self.version >= 10 else formats.FILE_HEADER
            handle.seek(0)
            self.header = header_type._make(struct.unpack(self.pack, handle.read(struct.calcsize(self.pack))))
            if self.header.num_programs != expected_count:
                raise ValueError("QPY program count differs from the saved plan")
            self.type_key = handle.read(1) if self.version >= 5 else b""
            if self.type_key and self.type_key != type_keys.Program.CIRCUIT:
                raise ValueError("QPY archive must contain circuits")
            self.offsets = None
            if self.version >= 16:
                self.offsets = [struct.unpack(formats.CIRCUIT_TABLE_ENTRY_PACK,
                    handle.read(formats.CIRCUIT_TABLE_ENTRY_SIZE))[0] for _ in range(expected_count)]
            self.payload_start = handle.tell()
        if self.offsets is not None and (not self.offsets or self.offsets[0] != self.payload_start
                or any(b <= a for a, b in zip(self.offsets, self.offsets[1:])) or self.offsets[-1] >= self.size):
            raise ValueError("invalid or truncated QPY circuit table")

    def _load(self, first, last, start, stop):
        parameters = inspect.signature(binary_io.read_circuit).parameters
        options = {"use_symengine": bool(self.version >= 10 and self.header.symbolic_encoding == type_keys.SymExprEncoding.SYMENGINE)}
        if "use_rust" in parameters:
            options["use_rust"] = False
        with self.path.open("rb") as handle:
            handle.seek(start)
            guarded = _ReadGuard(handle)
            programs = []
            for index in range(first, last):
                if self.offsets is not None:
                    handle.seek(self.offsets[index])
                programs.append(binary_io.read_circuit(guarded, self.version, **options))
                if self.offsets is not None:
                    end = self.offsets[index+1] if index+1 < self.header.num_programs else self.size
                    if handle.tell() != end:
                        raise ValueError("QPY payload has trailing or missing bytes")
            consumed = handle.tell()
        if len(programs) != last-first or consumed <= start or consumed > stop:
            raise ValueError("incomplete QPY circuit range")
        return programs, consumed

    def iter_batches(self, batches):
        position = self.payload_start
        previous = 0
        for batch in batches:
            first, last = batch["start"], batch["stop"]
            if first != previous or not first < last <= self.header.num_programs:
                raise ValueError("batch ranges must cover the QPY programs in order")
            start = self.offsets[first] if self.offsets is not None else position
            stop = (self.offsets[last] if last < self.header.num_programs else self.size) if self.offsets is not None else self.size
            programs, position = self._load(first, last, start, stop)
            if self.offsets is not None and position != stop:
                raise ValueError("QPY payload has trailing or missing bytes")
            if last == self.header.num_programs and position != self.size:
                raise ValueError("QPY archive has trailing bytes")
            yield programs, [start, position]
            del programs
            previous = last
        if previous != self.header.num_programs or position != self.size:
            raise ValueError("incomplete QPY coverage or trailing bytes")

    def load_batch(self, batch):
        first, last = batch["start"], batch["stop"]
        start, stop = batch["isa_byte_range"]
        if not self.payload_start <= start < stop <= self.size:
            raise ValueError("invalid saved QPY byte range")
        if self.offsets is not None:
            expected_stop = self.offsets[last] if last < self.header.num_programs else self.size
            if start != self.offsets[first] or stop != expected_stop:
                raise ValueError("saved QPY range differs from the archive table")
        programs, consumed = self._load(first, last, start, stop)
        if consumed != stop:
            raise ValueError("saved QPY range has trailing or missing bytes")
        return programs
