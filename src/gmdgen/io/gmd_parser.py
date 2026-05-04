# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path

from gmdgen.data.schema import GMDDocument, TagMap

_LEGACY_TAG_PATTERN = re.compile(
    r"<k>(.*?)</k>\s*(?:<([a-z])>(.*?)</\2>|<([a-z])\s*/>)",
    re.DOTALL,
)


def _parse_dict_like_node(node: ET.Element) -> TagMap:
    tags: TagMap = {}
    children = [child for child in list(node) if isinstance(child.tag, str)]

    index = 0
    while index < len(children):
        key_node = children[index]
        if key_node.tag != "k":
            index += 1
            continue

        key = (key_node.text or "").strip()
        if not key:
            index += 1
            continue

        if index + 1 >= len(children):
            break

        value_node = children[index + 1]
        if value_node.tag == "k":
            index += 1
            continue

        value_type = value_node.tag
        value = value_node.text or ""
        tags[key] = (value_type, value)
        index += 2

    return tags


def _parse_xml(raw_text: str) -> TagMap | None:
    try:
        root = ET.fromstring(raw_text.strip())
    except ET.ParseError:
        return None

    if root.tag == "plist":
        dict_node = root.find("dict")
        if dict_node is None:
            return {}
        return _parse_dict_like_node(dict_node)

    if root.tag in {"dict", "d"}:
        return _parse_dict_like_node(root)

    return None


def _parse_legacy_regex(raw_text: str) -> TagMap:
    tags: TagMap = {}
    for key, value_type_normal, value_normal, value_type_bool in _LEGACY_TAG_PATTERN.findall(
        raw_text
    ):
        value_type = value_type_normal or value_type_bool
        value = value_normal if value_type_normal else ""
        tags[unescape(key)] = (value_type, unescape(value))
    return tags


def parse_gmd_text(raw_text: str) -> TagMap:
    xml_tags = _parse_xml(raw_text)
    if xml_tags is not None:
        return xml_tags

    legacy_tags = _parse_legacy_regex(raw_text)
    if legacy_tags:
        return legacy_tags

    raise ValueError("No supported gmd key-value structure found.")


def parse_gmd_file(path: Path) -> GMDDocument:
    raw_text = path.read_text(encoding="utf-8-sig", errors="replace")
    tags = parse_gmd_text(raw_text)
    return GMDDocument(path=path, raw_text=raw_text, tags=tags)


def tags_to_plain_map(tags: TagMap) -> dict[str, str]:
    return {key: value for key, (_, value) in tags.items()}
