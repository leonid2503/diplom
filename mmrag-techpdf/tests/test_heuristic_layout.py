"""Tests for HeuristicLayoutAnalyzer._classify_text_block (pure heuristics, no PDF I/O)."""
import pytest

from mmrag.adapters.layout.heuristic_layout import HeuristicLayoutAnalyzer


@pytest.fixture
def analyzer():
    return HeuristicLayoutAnalyzer()


def _make_lines(n_lines, spans_per_line=3):
    """Build a minimal list-of-lines structure matching PyMuPDF's dict format."""
    return [{"spans": [{"text": f"word{i}"} for _ in range(spans_per_line)]} for i in range(n_lines)]


# ---------------------------------------------------------------------------
# _classify_text_block
# ---------------------------------------------------------------------------

class TestClassifyTextBlock:
    def test_normal_paragraph_is_text(self, analyzer):
        text = "This is a normal paragraph without tabs."
        lines = _make_lines(3, spans_per_line=5)
        assert analyzer._classify_text_block(text, lines) == "text"

    def test_many_tabs_triggers_table(self, analyzer):
        # More than 4 tab characters → table
        text = "col1\tcol2\tcol3\tcol4\tcol5"
        lines = _make_lines(2)
        assert analyzer._classify_text_block(text, lines) == "table"

    def test_exactly_four_tabs_is_text(self, analyzer):
        # 4 tabs is NOT > 4, so stays text
        text = "a\tb\tc\td"
        lines = _make_lines(2)
        assert analyzer._classify_text_block(text, lines) == "text"

    def test_five_tabs_is_table(self, analyzer):
        text = "a\tb\tc\td\te"
        lines = _make_lines(2)
        assert analyzer._classify_text_block(text, lines) == "table"

    def test_short_uniform_lines_is_table(self, analyzer):
        # avg spans per line ≤ 2 AND >= 4 lines → table
        text = "header row data\nrow1 col1 col2\nrow2 col1 col2\nrow3 col1 col2"
        lines = _make_lines(4, spans_per_line=1)  # 1 span/line → avg=1 ≤ 2
        assert analyzer._classify_text_block(text, lines) == "table"

    def test_few_lines_not_table_even_if_short(self, analyzer):
        # Only 3 lines → condition requires >= 4
        text = "a b\nc d\ne f"
        lines = _make_lines(3, spans_per_line=1)
        assert analyzer._classify_text_block(text, lines) == "text"

    def test_many_lines_with_rich_spans_is_text(self, analyzer):
        # avg > 2 → not table
        text = "long detailed sentence on each line " * 4
        lines = _make_lines(5, spans_per_line=5)  # avg=5 > 2
        assert analyzer._classify_text_block(text, lines) == "text"

    def test_empty_lines_list_is_text(self, analyzer):
        assert analyzer._classify_text_block("some text", []) == "text"

    def test_two_lines_with_few_spans_is_text(self, analyzer):
        # <= 2 lines → guard `len(lines) > 2` fails
        text = "col1 col2"
        lines = _make_lines(2, spans_per_line=1)
        assert analyzer._classify_text_block(text, lines) == "text"


# ---------------------------------------------------------------------------
# analyze_page raises NotImplementedError
# ---------------------------------------------------------------------------

class TestAnalyzePage:
    def test_raises_not_implemented(self, analyzer):
        with pytest.raises(NotImplementedError, match="analyze_document"):
            analyzer.analyze_page("some_image.png", page_number=1)
