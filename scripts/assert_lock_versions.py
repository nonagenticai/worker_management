"""Build-time guard: the image must run exactly the versions pinned in uv.lock.

Copied from nonagenticai/license-authz#63 (gitops#4248): an image that resolved
dependencies at build time shipped pyjwt 2.15.1 while uv.lock said 2.14.0, and
auth broke estate-wide. This Dockerfile used to run `uv lock` in the image too;
it now installs with `uv sync --frozen` from the committed lock, and this script
proves it.

Usage: python assert_lock_versions.py <uv.lock> <pins.txt>
  <pins.txt> is `uv export --frozen --no-dev --no-emit-project --no-hashes`.
Exits non-zero on any mismatch.
"""

import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version

# Asserted explicitly, whatever the pins file says (gitops#4248).
CRITICAL = ("pyjwt",)


def installed(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def lock_version(lock_path: str, name: str) -> str:
    with open(lock_path, "rb") as fh:
        lock = tomllib.load(fh)
    for pkg in lock.get("package", []):
        if pkg["name"] == name:
            return pkg["version"]
    raise SystemExit(f"FATAL: {name} not found in {lock_path}")


def main(lock_path: str, pins_path: str) -> int:
    errors: list[str] = []

    for name in CRITICAL:
        want = lock_version(lock_path, name)
        got = installed(name)
        if got != want:
            errors.append(f"{name}: installed {got} but uv.lock pins {want} (gitops#4248 guard)")
        else:
            print(f"OK: {name}=={got} matches uv.lock")

    checked = 0
    with open(pins_path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "==" not in line:
                continue
            spec, _, marker = line.partition(";")
            name, _, want = spec.strip().partition("==")
            name = name.split("[")[0].strip()
            want = want.strip()
            got = installed(name)
            if got is None and marker.strip():
                # Platform-conditional pin assumed not applicable to this image.
                # Only win32-only pins (colorama, tzdata) are expected to hit
                # this branch; cffi, pycparser and uvloop are conditional too
                # but evaluate true on CPython Linux and are installed, so they
                # are still checked above.
                continue
            checked += 1
            if got != want:
                errors.append(f"{name}: installed {got} but uv.lock pins {want}")

    if checked == 0:
        errors.append(f"checked 0 pins from {pins_path} - guard would pass vacuously")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(f"OK: {checked} installed packages match uv.lock")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
