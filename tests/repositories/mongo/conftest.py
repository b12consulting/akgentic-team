"""Skip-gate for the Mongo-specific test directory.

The actual ``mongo_client`` / ``mongo_db`` / ``mongo_store`` fixtures
live in ``tests/conftest.py`` so that the shared
``TestEventStoreContract`` parametrized suite and the migration-script
specs under ``tests/scripts/`` can both compose them. Tests
under this directory inherit those fixtures through normal pytest
fixture resolution. This module only exists to skip the entire directory
cleanly when ``pymongo`` or ``mongomock`` is unavailable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymongo")
pytest.importorskip("mongomock")
