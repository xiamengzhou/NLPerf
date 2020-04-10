import pytest

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.collect_feats import build_index


@pytest.mark.unit
def test_file_index(main_path):
    index = build_index(main_dir=main_path, subdir_name="tok")
    assert len(index) > 0
    assert "WikiMatrix.ar-ru.txt.ru.tok" in index


