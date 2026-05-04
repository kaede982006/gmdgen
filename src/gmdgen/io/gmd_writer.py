from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from gmdgen.data.schema import TagMap


def serialize_tags(tags: TagMap) -> str:
    plist = ET.Element("plist", {"version": "1.0", "gjver": "2.0"})
    dict_node = ET.SubElement(plist, "dict")

    for key, (value_type, value) in tags.items():
        key_node = ET.SubElement(dict_node, "k")
        key_node.text = key

        value_node = ET.SubElement(dict_node, value_type)
        if value_type not in {"t", "f"}:
            value_node.text = value

    xml_body = ET.tostring(plist, encoding="unicode", short_empty_elements=True)
    return f'<?xml version="1.0"?>{xml_body}'


def write_gmd_file(path: Path, tags: TagMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_tags(tags), encoding="utf-8")
