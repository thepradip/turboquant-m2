# Step-by-Step Guide: Create & Publish a Python Package to TestPyPI and PyPI

This guide walks through every step — from an empty folder to a published `pip install`-able package — using `turboquant` as a real example.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Write Your Code](#3-write-your-code)
4. [Create pyproject.toml](#4-create-pyprojecttoml)
5. [Create README.md](#5-create-readmemd)
6. [Add a LICENSE](#6-add-a-license)
7. [Write Tests](#7-write-tests)
8. [Run Tests & Quality Checks](#8-run-tests--quality-checks)
9. [Build the Package](#9-build-the-package)
10. [Publish to TestPyPI](#10-publish-to-testpypi)
11. [Test Install from TestPyPI](#11-test-install-from-testpypi)
12. [Publish to Real PyPI](#12-publish-to-real-pypi)
13. [Verify on PyPI](#13-verify-on-pypi)
14. [Updating Your Package](#14-updating-your-package)
15. [Common Errors & Fixes](#15-common-errors--fixes)

---

## 1. Prerequisites

### Install required tools

```bash
# Build tools
pip install build twine

# Development tools (optional but recommended)
pip install pytest ruff bandit
```

### Create accounts

| Service | URL | Purpose |
|---------|-----|---------|
| **TestPyPI** | https://test.pypi.org/account/register/ | Testing uploads |
| **PyPI** | https://pypi.org/account/register/ | Real production uploads |

> **Important**: These are SEPARATE accounts. You need to register on both.

### Generate API tokens

**TestPyPI token:**
1. Go to https://test.pypi.org/manage/account/token/
2. Click "Add API token"
3. Token name: `turboquant-upload` (any name)
4. Scope: "Entire account" (for first upload; later restrict to project)
5. Copy the token (starts with `pypi-...`) — you only see it once!

**PyPI token:**
1. Go to https://pypi.org/manage/account/token/
2. Same steps as above
3. Copy and save the token securely

### (Optional) Save tokens in ~/.pypirc

Instead of pasting tokens every time, create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_REAL_PYPI_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_PYPI_TOKEN_HERE
```

Then secure the file:
```bash
chmod 600 ~/.pypirc
```

> **Security**: Never commit `.pypirc` to git. Add it to your global `.gitignore`.

---

## 2. Project Structure

Create the standard `src` layout:

```bash
mkdir -p mypackage/src/mypackage
mkdir -p mypackage/tests
cd mypackage
```

Final structure should look like:

```
mypackage/
├── pyproject.toml        # Package metadata & build config
├── README.md             # Package description (shown on PyPI)
├── LICENSE               # Open source license
├── src/
│   └── mypackage/
│       ├── __init__.py   # Public API + version
│       ├── core.py       # Your main code
│       └── utils.py      # Helper modules
└── tests/
    ├── __init__.py
    ├── test_core.py
    └── test_utils.py
```

**Why `src/` layout?**
- Prevents accidentally importing from local directory instead of installed package
- Ensures your tests run against the *installed* package, not source files
- Industry standard (recommended by PyPA)

---

## 3. Write Your Code

### src/mypackage/__init__.py

```python
"""My awesome package."""

__version__ = "0.1.0"

from .core import MyMainClass, my_function

__all__ = ["MyMainClass", "my_function", "__version__"]
```

**Key rules:**
- `__version__` — single source of truth for version
- `__all__` — controls what `from mypackage import *` exposes
- Only import public API here; keep internals private

### src/mypackage/core.py

```python
"""Core functionality."""

def my_function(x):
    """Does something useful."""
    return x * 2

class MyMainClass:
    """Main class for doing things."""

    def __init__(self, config: int = 10):
        if config < 1:
            raise ValueError(f"config must be >= 1, got {config}")
        self.config = config

    def process(self, data):
        return data * self.config
```

---

## 4. Create pyproject.toml

This is the **single config file** that replaces `setup.py`, `setup.cfg`, and `MANIFEST.in`.

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "your-package-name"        # Must be unique on PyPI!
version = "0.1.0"                 # Semantic versioning: MAJOR.MINOR.PATCH
description = "One-line description of your package"
readme = "README.md"
license = "MIT"                   # SPDX identifier
requires-python = ">=3.9"
authors = [
    {name = "Your Name", email = "you@example.com"},
]
keywords = ["keyword1", "keyword2"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]

# Required dependencies (installed automatically)
dependencies = [
    "numpy>=1.24.0",
    "requests>=2.28.0",
]

# Optional dependencies (installed with pip install mypackage[extra])
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.1.0",
    "bandit>=1.7",
]

[project.urls]
Homepage = "https://github.com/you/mypackage"
Repository = "https://github.com/you/mypackage"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### Version numbering guide:

| Version | When to use |
|---------|-------------|
| `0.1.0` | First release, experimental |
| `0.1.1` | Bug fix |
| `0.2.0` | New feature, backward compatible |
| `1.0.0` | Stable public API |
| `2.0.0` | Breaking changes |

---

## 5. Create README.md

PyPI renders this as your project page. Include:

```markdown
# My Package

One paragraph explaining what it does.

## Install

\```bash
pip install mypackage
\```

## Quick Start

\```python
from mypackage import MyMainClass

obj = MyMainClass(config=5)
result = obj.process(10)  # returns 50
\```

## License

MIT
```

> **Tip**: PyPI supports full GitHub-flavored Markdown — tables, code blocks, badges all work.

---

## 6. Add a LICENSE

Choose a license at https://choosealicense.com/ and copy the full text to `LICENSE`.

Common choices:
| License | Use case |
|---------|----------|
| MIT | Maximum freedom, minimal restrictions |
| Apache-2.0 | Like MIT but with patent protection |
| GPL-3.0 | Requires derivative works to also be open source |

---

## 7. Write Tests

### tests/__init__.py
```python
# Empty file — makes tests/ a package
```

### tests/test_core.py
```python
import pytest
from mypackage import MyMainClass, my_function

class TestMyFunction:
    def test_basic(self):
        assert my_function(5) == 10

    def test_zero(self):
        assert my_function(0) == 0

    def test_negative(self):
        assert my_function(-3) == -6

class TestMyMainClass:
    def test_init(self):
        obj = MyMainClass(config=5)
        assert obj.config == 5

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            MyMainClass(config=0)

    def test_process(self):
        obj = MyMainClass(config=3)
        assert obj.process(10) == 30
```

---

## 8. Run Tests & Quality Checks

### Step 8a: Install package in dev mode

```bash
cd mypackage/
pip install -e ".[dev]"
```

The `-e` flag = "editable" — changes to source code are reflected immediately without reinstalling.

### Step 8b: Run tests

```bash
pytest -v
```

Expected output:
```
tests/test_core.py::TestMyFunction::test_basic PASSED
tests/test_core.py::TestMyFunction::test_zero PASSED
tests/test_core.py::TestMyFunction::test_negative PASSED
tests/test_core.py::TestMyMainClass::test_init PASSED
tests/test_core.py::TestMyMainClass::test_invalid_config PASSED
tests/test_core.py::TestMyMainClass::test_process PASSED

6 passed
```

> **DO NOT proceed to publishing until ALL tests pass.**

### Step 8c: Lint check

```bash
ruff check src/
```

Fix any issues:
```bash
ruff check src/ --fix
```

### Step 8d: Security scan

```bash
bandit -r src/ -ll
```

Should show: `No issues identified.`

---

## 9. Build the Package

```bash
# Clean previous builds
rm -rf dist/ build/ src/*.egg-info

# Build both sdist (.tar.gz) and wheel (.whl)
python -m build
```

This creates:
```
dist/
├── mypackage-0.1.0-py3-none-any.whl    # Wheel (preferred, fast install)
└── mypackage-0.1.0.tar.gz              # Source distribution (fallback)
```

### Verify the build

```bash
twine check dist/*
```

Must show:
```
Checking dist/mypackage-0.1.0-py3-none-any.whl: PASSED
Checking dist/mypackage-0.1.0.tar.gz: PASSED
```

### (Optional) Test install locally

```bash
# Create temporary venv
python -m venv /tmp/test_install
source /tmp/test_install/bin/activate

# Install from wheel
pip install dist/mypackage-0.1.0-py3-none-any.whl

# Test it works
python -c "from mypackage import __version__; print(f'v{__version__} works!')"

# Cleanup
deactivate
rm -rf /tmp/test_install
```

---

## 10. Publish to TestPyPI

TestPyPI is a sandbox — use it to verify everything works before publishing for real.

### Option A: Using ~/.pypirc (if configured)

```bash
twine upload --repository testpypi dist/*
```

### Option B: Using token directly

```bash
twine upload --repository testpypi dist/* \
  --username __token__ \
  --password pypi-YOUR_TEST_TOKEN_HERE
```

### Option C: Using environment variables

```bash
TWINE_USERNAME=__token__ \
TWINE_PASSWORD=pypi-YOUR_TEST_TOKEN_HERE \
twine upload --repository testpypi dist/*
```

### Expected output

```
Uploading distributions to https://test.pypi.org/legacy/
Uploading mypackage-0.1.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 31.2/31.2 kB
Uploading mypackage-0.1.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 33.3/33.3 kB

View at:
https://test.pypi.org/project/mypackage/0.1.0/
```

---

## 11. Test Install from TestPyPI

```bash
# Create fresh venv
python -m venv /tmp/testpypi_check
source /tmp/testpypi_check/bin/activate

# Install from TestPyPI
# --extra-index-url lets it find dependencies (numpy, etc.) from real PyPI
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  mypackage

# Verify
python -c "
from mypackage import __version__, MyMainClass
print(f'Installed v{__version__} from TestPyPI')
obj = MyMainClass(5)
print(f'process(10) = {obj.process(10)}')
print('SUCCESS!')
"

# Cleanup
deactivate
rm -rf /tmp/testpypi_check
```

> **If this works, you're ready for real PyPI!**

---

## 12. Publish to Real PyPI

### Option A: Using ~/.pypirc

```bash
twine upload dist/*
```

### Option B: Using token directly

```bash
twine upload dist/* \
  --username __token__ \
  --password pypi-YOUR_REAL_PYPI_TOKEN_HERE
```

### Option C: Using environment variables

```bash
TWINE_USERNAME=__token__ \
TWINE_PASSWORD=pypi-YOUR_REAL_PYPI_TOKEN_HERE \
twine upload dist/*
```

### Expected output

```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading mypackage-0.1.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 31.2/31.2 kB
Uploading mypackage-0.1.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 33.3/33.3 kB

View at:
https://pypi.org/project/mypackage/0.1.0/
```

---

## 13. Verify on PyPI

### Check the project page
Visit: `https://pypi.org/project/mypackage/`

Verify:
- [ ] README renders correctly
- [ ] Version is correct
- [ ] Dependencies listed
- [ ] License shown
- [ ] Author info correct

### Test install from PyPI

```bash
pip install mypackage
python -c "from mypackage import __version__; print(__version__)"
```

---

## 14. Updating Your Package

### Step 1: Bump the version

Edit `pyproject.toml`:
```toml
version = "0.2.0"  # was 0.1.0
```

Edit `src/mypackage/__init__.py`:
```python
__version__ = "0.2.0"  # was 0.1.0
```

> **Tip**: Keep version in sync between both files. For automation, consider `setuptools-scm` or `bump2version`.

### Step 2: Clean, build, check, upload

```bash
# Clean old builds
rm -rf dist/ build/ src/*.egg-info

# Build
python -m build

# Verify
twine check dist/*

# Upload (TestPyPI first, then PyPI)
twine upload --repository testpypi dist/*
twine upload dist/*
```

> **You CANNOT re-upload the same version.** Even if you delete a release, the version number is permanently reserved. Always bump the version.

---

## 15. Common Errors & Fixes

### "File already exists"
```
HTTPError: 400 Bad Request
File already exists.
```
**Fix**: Bump the version number. You cannot re-upload the same version, ever.

### "Invalid or non-existent authentication"
```
HTTPError: 403 Forbidden
Invalid or non-existent authentication information.
```
**Fix**:
- Ensure username is exactly `__token__` (two underscores each side)
- Ensure the token starts with `pypi-`
- TestPyPI tokens don't work on PyPI and vice versa

### "README rendering fails"
```
The description failed to render for 'text/markdown'
```
**Fix**:
- Ensure `readme = "README.md"` is in pyproject.toml
- Validate locally: `pip install readme-renderer && python -m readme_renderer README.md`

### "Package name taken"
```
HTTPError: 400 Bad Request
The name 'mypackage' is too similar to an existing project.
```
**Fix**: Choose a different, more unique package name.

### "No module named mypackage"
After `pip install`, import fails.
**Fix**: Ensure `[tool.setuptools.packages.find] where = ["src"]` is in pyproject.toml.

### "twine: command not found"
**Fix**: `pip install twine`

---

## Quick Reference: Full Command Sequence

```bash
# === ONE-TIME SETUP ===
pip install build twine pytest ruff bandit
# Register at https://test.pypi.org and https://pypi.org
# Generate API tokens on both

# === DEVELOP ===
pip install -e ".[dev]"      # Install in editable mode
pytest -v                     # Run tests
ruff check src/               # Lint
bandit -r src/ -ll            # Security scan

# === BUILD ===
rm -rf dist/ build/ src/*.egg-info
python -m build
twine check dist/*

# === PUBLISH TO TESTPYPI ===
twine upload --repository testpypi dist/*

# === VERIFY ON TESTPYPI ===
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ mypackage

# === PUBLISH TO REAL PYPI ===
twine upload dist/*

# === VERIFY ON PYPI ===
pip install mypackage
```

---

## Security Best Practices

1. **Never commit tokens to git** — use `~/.pypirc` or environment variables
2. **Use project-scoped tokens** — after first upload, create a token restricted to your project only
3. **Enable 2FA on PyPI** — https://pypi.org/manage/account/two-factor/
4. **Never share tokens in chat/email** — if exposed, revoke immediately and create a new one
5. **Add `.pypirc` to global gitignore**:
   ```bash
   echo ".pypirc" >> ~/.gitignore_global
   git config --global core.excludesfile ~/.gitignore_global
   ```

---

*This guide was created while publishing the `turboquant` package. See the actual published result at https://test.pypi.org/project/turboquant/0.1.0/*
