from __future__ import annotations

import hashlib
import io
import re
import stat
import zipfile
from pathlib import PurePosixPath
from typing import Iterable

from .languages import SUPPORTED_LANGUAGES, detect_language
from .models import InputFile, Submission

MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_EXTRACTED_BYTES = 500 * 1024 * 1024
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_FILES = 500

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


class IngestionError(ValueError):
    """Raised when an uploaded file cannot be safely ingested."""


def _student_id(name: str, fallback_index: int) -> str:
    stem = PurePosixPath(name).stem
    clean = _SAFE_ID.sub("_", stem).strip("._-")
    return clean or f"student_{fallback_index:03d}"


def _validate_zip_member(info: zipfile.ZipInfo) -> None:
    name = info.filename.replace("\\", "/")
    path = PurePosixPath(name)

    if not name or path.is_absolute() or ".." in path.parts:
        raise IngestionError(f"unsafe archive path: {info.filename!r}")

    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise IngestionError(f"symlink-like archive entry rejected: {info.filename!r}")

    if info.is_dir():
        return

    if name.lower().endswith(".zip"):
        raise IngestionError("nested ZIP archives are not allowed")

    if PurePosixPath(name).suffix.lower() not in SUPPORTED_LANGUAGES:
        return

    if info.file_size > MAX_FILE_BYTES:
        raise IngestionError(f"source file exceeds limit: {info.filename!r}")


def _read_zip(upload: bytes) -> list[tuple[str, bytes]]:
    if len(upload) > MAX_ARCHIVE_BYTES:
        raise IngestionError("ZIP archive exceeds the upload size limit")

    try:
        archive = zipfile.ZipFile(io.BytesIO(upload))
    except zipfile.BadZipFile as exc:
        raise IngestionError("invalid ZIP archive") from exc

    entries = [info for info in archive.infolist() if not info.is_dir()]
    if len(entries) > MAX_FILES:
        raise IngestionError("archive contains too many files")

    extracted_total = 0
    result: list[tuple[str, bytes]] = []

    with archive:
        for info in entries:
            _validate_zip_member(info)
            if PurePosixPath(info.filename).suffix.lower() not in SUPPORTED_LANGUAGES:
                continue

            extracted_total += info.file_size
            if extracted_total > MAX_EXTRACTED_BYTES:
                raise IngestionError("archive expands beyond the extraction limit")

            data = archive.read(info)
            if len(data) != info.file_size:
                raise IngestionError(f"unexpected archive read size: {info.filename!r}")
            result.append((info.filename, data))

    return result


def _decode_source(data: bytes, filename: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionError(f"source file is not valid UTF-8: {filename!r}") from exc


def ingest(inputs: Iterable[InputFile]) -> list[Submission]:
    files: list[tuple[str, bytes]] = []

    for upload in inputs:
        if len(upload.data) > MAX_ARCHIVE_BYTES:
            raise IngestionError(f"uploaded file is too large: {upload.name!r}")

        if upload.name.lower().endswith(".zip"):
            files.extend(_read_zip(upload.data))
        elif PurePosixPath(upload.name).suffix.lower() in SUPPORTED_LANGUAGES:
            if len(upload.data) > MAX_FILE_BYTES:
                raise IngestionError(f"source file exceeds limit: {upload.name!r}")
            files.append((upload.name, upload.data))
        else:
            supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
            raise IngestionError(f"unsupported input type: {upload.name!r}. Supported extensions: {supported}")

    if not files:
        raise IngestionError("no supported source submissions were found")
    if len(files) > MAX_FILES:
        raise IngestionError("too many source submissions")

    submissions: list[Submission] = []
    seen_ids: set[str] = set()

    for index, (name, data) in enumerate(files, start=1):
        student_id = _student_id(name, index)
        if student_id in seen_ids:
            student_id = f"{student_id}_{index:03d}"
        seen_ids.add(student_id)

        source = _decode_source(data, name)
        source_hash = hashlib.sha256(data).hexdigest()
        try:
            language = detect_language(name)
        except ValueError as exc:
            raise IngestionError(str(exc)) from exc

        submissions.append(
            Submission(
                student_id=student_id,
                filename=name,
                source=source,
                source_hash=source_hash,
                language=language,
            )
        )

    return submissions
