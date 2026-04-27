from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

from .chunking import RecursiveTextChunker
from .models import PolicyChunk, PolicyDocument

NAMESPACE = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _normalize(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _read_shared_strings(zip_file: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return []
    root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    shared_strings: list[str] = []
    for item in root.findall("a:si", NAMESPACE):
        parts: list[str] = []
        for node in item.iterfind(".//a:t", NAMESPACE):
            if node.text:
                parts.append(node.text)
        shared_strings.append("".join(parts))
    return shared_strings


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("a:v", NAMESPACE)
    if cell_type == "s" and value_node is not None and value_node.text is not None:
        return shared_strings[int(value_node.text)]
    if cell_type == "inlineStr":
        inline = cell.find("a:is", NAMESPACE)
        if inline is not None:
            text_node = inline.find(".//a:t", NAMESPACE)
            return text_node.text if text_node is not None and text_node.text is not None else ""
    return value_node.text if value_node is not None and value_node.text is not None else ""


def load_policy_documents(dataset_path: Path) -> list[PolicyDocument]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with ZipFile(dataset_path) as zip_file:
        shared_strings = _read_shared_strings(zip_file)
        sheet_root = ET.fromstring(zip_file.read("xl/worksheets/sheet1.xml"))

    rows = sheet_root.findall(".//a:sheetData/a:row", NAMESPACE)
    headers: list[str] = []
    documents: list[PolicyDocument] = []
    for row_index, row in enumerate(rows):
        values = {}
        for cell in row.findall("a:c", NAMESPACE):
            reference = cell.attrib.get("r", "")
            column = re.sub(r"\d+", "", reference)
            values[column] = _cell_value(cell, shared_strings)

        if row_index == 0:
            headers = [_normalize(values.get(col, "")) for col in ["A", "B", "C", "D", "E"]]
            continue

        trouble = _normalize(values.get("A"))
        if not trouble:
            continue
        category = _normalize(values.get("B"))
        solution = _normalize(values.get("C"))
        alternate = _normalize(values.get("D"))
        company_response = _normalize(values.get("E"))
        content = " | ".join(
            part
            for part in [
                f"Trouble: {trouble}",
                f"Category: {category}" if category else "",
                f"Solution: {solution}" if solution else "",
                f"Alternate Solution: {alternate}" if alternate else "",
                f"Company Response: {company_response}" if company_response else "",
            ]
            if part
        )
        documents.append(
            PolicyDocument(
                id=f"policy-{row_index}",
                title=trouble,
                category=category,
                solution=solution,
                alternate_solution=alternate,
                company_response=company_response,
                content=content,
            )
        )

    return documents


def load_policy_chunks(dataset_path: Path, chunk_size: int = 500, chunk_overlap: int = 80) -> list[PolicyChunk]:
    documents = load_policy_documents(dataset_path)
    chunker = RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[PolicyChunk] = []

    for document in documents:
        fragments = chunker.split_text(document.content) or [document.content]
        fragment_count = len(fragments)
        for chunk_index, fragment in enumerate(fragments, start=1):
            chunks.append(
                PolicyChunk(
                    id=f"{document.id}-chunk-{chunk_index}",
                    source_id=document.id,
                    title=document.title,
                    category=document.category,
                    solution=document.solution,
                    alternate_solution=document.alternate_solution,
                    company_response=document.company_response,
                    content=fragment,
                    chunk_index=chunk_index,
                    chunk_count=fragment_count,
                )
            )

    return chunks


@lru_cache(maxsize=1)
def cached_policy_documents(dataset_path_str: str) -> tuple[PolicyDocument, ...]:
    return tuple(load_policy_documents(Path(dataset_path_str)))


@lru_cache(maxsize=4)
def cached_policy_chunks(dataset_path_str: str, chunk_size: int = 500, chunk_overlap: int = 80) -> tuple[PolicyChunk, ...]:
    return tuple(load_policy_chunks(Path(dataset_path_str), chunk_size=chunk_size, chunk_overlap=chunk_overlap))
