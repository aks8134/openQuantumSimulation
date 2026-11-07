"""Create a portable ZIP of a folder using only the Python standard library.

Example (PowerShell, from the repository root):
    python zip_folder.py "Model2/Experiment7/hardware_runs/RUN_NAME" --exclude "*.qpy" "*.html" "*.partial"

The default destination is next to the folder: RUN_NAME.zip. Relative paths
and the top-level folder name are preserved. Files are streamed into the ZIP;
source files are never modified. Links/junctions are not followed.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile


def is_link(path):
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def excluded(path, source, patterns):
    relative = path.relative_to(source).as_posix()
    return any(fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern)
               for pattern in patterns)


def raise_walk_error(error):
    raise error


def zip_folder(folder, *, output=None, exclude=(), overwrite=False):
    source = Path(folder).expanduser().resolve(strict=True)
    if not source.is_dir():
        raise ValueError(f"not a folder: {source}")
    destination = (Path(output).expanduser().resolve() if output is not None
                   else source.parent/(source.name+".zip"))
    if destination == source or destination.is_dir():
        raise ValueError("output must be a ZIP filename, not a folder")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"archive already exists: {destination}; use --output with a new name or --overwrite")

    # Plan before creating the temporary ZIP so an output inside the source
    # folder cannot recursively include itself. Do not silently skip IO errors.
    files, empty_directories = [], []
    for current, directories, filenames in os.walk(source, followlinks=False, onerror=raise_walk_error):
        current = Path(current)
        directories[:] = sorted(name for name in directories
            if not is_link(current/name) and not excluded(current/name, source, exclude))
        included = []
        for name in sorted(filenames):
            path = current/name
            if path.resolve() == destination or is_link(path) or excluded(path, source, exclude):
                continue
            if not path.is_file():
                raise ValueError(f"unsupported non-regular file: {path}")
            included.append(path)
        files.extend(included)
        if not directories and not included:
            empty_directories.append(current)

    size = sum(path.stat().st_size for path in files)
    print(f"Packing {len(files):,} files ({size/1024**2:.1f} MiB before compression)", flush=True)
    print(f"Output: {destination}", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".zip_folder_", suffix=".partial", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, mode="w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            archive.writestr(source.name+"/", b"")
            for directory in empty_directories:
                if directory != source:
                    archive.writestr(source.name+"/"+directory.relative_to(source).as_posix()+"/", b"")
            for index, path in enumerate(files, 1):
                relative = path.relative_to(source).as_posix()
                archive.write(path, arcname=source.name+"/"+relative)
                if index % 25 == 0 or index == len(files):
                    print(f"Added {index:,}/{len(files):,}: {relative}", flush=True)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"archive appeared while packing; existing file retained: {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Created: {destination} ({destination.stat().st_size/1024**2:.1f} MiB)", flush=True)
    return destination


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder to package, including nested folders")
    parser.add_argument("--output", type=Path, help="ZIP filename; default: next to the input folder")
    parser.add_argument("--exclude", nargs="+", action="extend", default=[], help="quoted glob patterns to omit anywhere in the folder")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing destination ZIP")
    options = parser.parse_args(arguments)
    return zip_folder(options.folder, output=options.output, exclude=options.exclude, overwrite=options.overwrite)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
