"""前端静态资源一致性测试。

这类测试的价值在于：改了 HTML 却没改 JS（或反过来）时能立刻发现，
比人工点页面可靠。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "templates" / "index.html"
SCRIPT = REPO_ROOT / "static" / "app.js"
STYLE = REPO_ROOT / "static" / "style.css"


@pytest.fixture(scope="module")
def html() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js() -> str:
    return SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css() -> str:
    return STYLE.read_text(encoding="utf-8")


def test_frontend_files_exist():
    for path in (TEMPLATE, SCRIPT, STYLE):
        assert path.is_file() and path.stat().st_size > 0


def test_every_element_id_used_by_js_exists_in_html(html, js):
    used = set(re.findall(r'getElementById\("([^"]+)"\)', js))
    defined = set(re.findall(r'id="([^"]+)"', html))
    missing = sorted(used - defined)
    assert not missing, f"JS 引用了 HTML 中不存在的元素：{missing}"


def test_every_css_class_used_by_js_exists_in_stylesheet(js, css):
    """JS 动态拼出来的类名必须有对应样式，否则结果列表会走样。"""
    dynamic = set(re.findall(r'class="([a-z][a-z0-9-]+)"', js))
    missing = sorted(name for name in dynamic if f".{name}" not in css)
    assert not missing, f"样式表缺少这些类：{missing}"


def test_model_selector_is_wired(html, js):
    assert 'name="model"' in html, "模型下拉必须以 model 为字段名提交"
    assert "modelSelect" in js
    assert "/api/classes" in js, "切换模型时应重新拉取类别"


def test_class_picker_renders_from_template(html):
    assert "class-option" in html
    assert 'data-class=' in html


def test_js_has_no_placeholder_leftovers(js):
    for marker in ("TODO", "FIXME", "console.log(\"test"):
        assert marker not in js


def test_js_is_wrapped_in_iife(js):
    """脚本以 IIFE 包裹，避免把内部变量泄漏到全局。"""
    assert js.lstrip().startswith("(()"), "app.js 应以 IIFE 形式组织"
    assert js.rstrip().endswith("})();") or js.rstrip().endswith("})()"), "IIFE 必须闭合"


def test_template_has_no_leaked_debug_output(html):
    assert "console.log" not in html
    assert "{{ debug" not in html and "<<<<" not in html


def test_css_defines_variables_and_balanced_braces(css):
    assert css.count("--") > 0, "样式应使用 CSS 变量"
    assert css.count("{") == css.count("}"), "大括号必须成对"
