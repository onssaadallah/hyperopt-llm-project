import os
import pytest
from config_file import create_file, create_project_structure

def test_create_file(tmp_path):
    d = tmp_path / "subdir"
    p = d / "hello.txt"
    create_file(str(p), "content")
    assert p.read_text(encoding="utf-8") == "content"

def test_create_project_structure(tmp_path):
    base_dir = tmp_path / "test_project"
    create_project_structure(str(base_dir))
    
    assert (base_dir / "src").is_dir()
    assert (base_dir / "src/__init__.py").exists()
    assert (base_dir / "data/raw_data").is_dir()
