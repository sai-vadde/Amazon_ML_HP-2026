from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from zipfile import ZipFile

EXPECTED_SHA256 = "2AEFD2F8EB6F132B8933FCCC1CBB98FA6356756A7581FED5914A5F26704A8BC5"
DATA_FILES = {
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv",
}
REFERENCE_FILES = {
    "README.md",
    "Documentation_template.md",
    "validate_submission.py",
}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def extract_member(archive: ZipFile, member: str, destination: Path) -> None:
    info = archive.getinfo(member)
    if destination.exists():
        if destination.stat().st_size != info.file_size:
            raise RuntimeError(f"existing file has unexpected size: {destination}")
        print(f"verified existing {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with archive.open(info) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    temporary.replace(destination)
    print(f"extracted {destination.name} ({info.file_size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    archive_path = args.archive.resolve()
    project_root = args.project_root.resolve()
    actual_hash = sha256(archive_path)
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(f"archive SHA-256 mismatch: {actual_hash}")

    with ZipFile(archive_path) as archive:
        members: dict[str, str] = {}
        for name in archive.namelist():
            base = Path(name).name
            if base in DATA_FILES | REFERENCE_FILES and not name.startswith("__MACOSX/"):
                if base in members:
                    raise RuntimeError(f"duplicate archive member basename: {base}")
                members[base] = name
        missing = (DATA_FILES | REFERENCE_FILES) - set(members)
        if missing:
            raise RuntimeError(f"archive missing expected files: {sorted(missing)}")
        for base in sorted(DATA_FILES):
            extract_member(archive, members[base], project_root / "data" / "raw" / base)
        for base in sorted(REFERENCE_FILES):
            extract_member(archive, members[base], project_root / "organizer" / base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

