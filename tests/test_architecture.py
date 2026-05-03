import ast
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1] / "app"


def _app_imports(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names if alias.name.startswith("app."))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app."):
            imports.add(node.module)
    return imports


def test_domain_has_no_framework_or_adapter_dependencies():
    forbidden_prefixes = (
        "app.adapters",
        "app.application",
        "app.auth",
        "app.config",
        "app.db",
        "app.dependencies",
        "app.routers",
        "app.services",
    )

    violations = []
    for path in (APP_ROOT / "domain").rglob("*.py"):
        for imported in _app_imports(path):
            if imported.startswith(forbidden_prefixes):
                violations.append(f"{path.relative_to(APP_ROOT)} imports {imported}")

    assert violations == []


def test_application_has_no_adapter_dependencies():
    forbidden_prefixes = (
        "app.adapters",
        "app.auth",
        "app.config",
        "app.db",
        "app.dependencies",
        "app.routers",
        "app.services",
    )

    violations = []
    for path in (APP_ROOT / "application").rglob("*.py"):
        for imported in _app_imports(path):
            if imported.startswith(forbidden_prefixes):
                violations.append(f"{path.relative_to(APP_ROOT)} imports {imported}")

    assert violations == []
