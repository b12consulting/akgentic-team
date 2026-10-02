"""The shared three-key builder behind every backend's ``update_team_description``.

Backend behaviour lives in ``test_event_store_contract.py``. What is pinned
here is the one property the contract suite cannot see directly: the values
every backend writes are the values ``save_team`` would have written for the
same ``Process``, byte for byte, so no backend can hand-format a stamp its own
``load_team`` then refuses.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from akgentic.team.models import DescriptionOrigin, Process
from akgentic.team.repositories._description import (
    DESCRIPTION_UPDATE_KEYS,
    description_update_fields,
)
from tests.models.conftest import make_process

_MILLISECOND = timedelta(milliseconds=1)


class TestDescriptionUpdateFields:
    @pytest.mark.parametrize("origin", list(DescriptionOrigin))
    def test_the_keys_are_exactly_the_three(self, origin: DescriptionOrigin) -> None:
        fields = description_update_fields("A description", origin)

        assert tuple(fields) == DESCRIPTION_UPDATE_KEYS
        assert set(fields) == {"team_description", "description_origin", "updated_at"}

    @pytest.mark.parametrize("origin", list(DescriptionOrigin))
    @pytest.mark.parametrize("description", ["Quarterly review", None, "  padded  "])
    def test_the_values_match_a_save_team_style_dump_of_the_same_process(
        self, origin: DescriptionOrigin, description: str | None
    ) -> None:
        """Pick the same three keys out of a full ``Process`` dump: they must agree.

        ``updated_at`` is stamped inside the helper, so it is compared through
        the dump's own serialisation rather than by value: the dumped form must
        round-trip through ``Process.model_validate`` to a tz-aware stamp inside
        the call window.
        """
        before = datetime.now(UTC)
        fields = description_update_fields(description, origin)
        after = datetime.now(UTC)

        full = (
            make_process()
            .model_copy(
                update={
                    "team_description": description,
                    "description_origin": origin,
                    "updated_at": before,
                }
            )
            .model_dump()
        )
        assert fields["team_description"] == full["team_description"]
        assert fields["description_origin"] == full["description_origin"]
        assert type(fields["updated_at"]) is type(full["updated_at"])

        # The stamp is in the form the reader validates, and it is "now".
        restored = Process.model_validate({**full, "updated_at": fields["updated_at"]})
        assert restored.updated_at.tzinfo is not None
        assert before - _MILLISECOND <= restored.updated_at <= after + _MILLISECOND

    def test_the_origin_is_the_plain_stored_string_not_the_enum_member(self) -> None:
        """What reaches YAML / BSON / JSON is what ``save_team`` writes: the value."""
        fields = description_update_fields(None, DescriptionOrigin.USER)

        assert fields["description_origin"] == "user"
        assert type(fields["description_origin"]) is str

    def test_the_description_is_stored_verbatim(self) -> None:
        padded = "  " + "x" * 600 + "  "

        assert description_update_fields(padded, DescriptionOrigin.AUTO)["team_description"] == (
            padded
        )
