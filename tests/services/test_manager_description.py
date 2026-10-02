"""``TeamManager.update_description`` is a pass-through to the store, and nothing more.

The ownership guard lives in the store's own filter — that is what lets two
processes write without racing — so the one thing this facade must NOT do is
read, check and decide on its own. These specs pin the pass-through shape: the
store method is called exactly once with the arguments handed in, its answer is
returned unchanged, and no other store read happens on the way.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from akgentic.team.manager import TeamManager
from akgentic.team.models import DescriptionOrigin, Process
from tests.models.conftest import make_process
from tests.services.conftest import InMemoryEventStore


class RecordingDescriptionStore(InMemoryEventStore):
    """``InMemoryEventStore`` that records the arguments of every description write.

    The end state cannot show whether the manager called the store once with
    the caller's arguments or loaded, decided and wrote on its own — only the
    recorded calls can.
    """

    def __init__(self) -> None:
        super().__init__()
        self.description_calls: list[tuple[uuid.UUID, str | None, DescriptionOrigin]] = []
        self.load_team_calls = 0

    def update_team_description(
        self,
        team_id: uuid.UUID,
        description: str | None,
        origin: DescriptionOrigin,
    ) -> Process | None:
        self.description_calls.append((team_id, description, origin))
        return super().update_team_description(team_id, description, origin)

    def load_team(self, team_id: uuid.UUID) -> Process | None:
        self.load_team_calls += 1
        return super().load_team(team_id)


@pytest.fixture()
def event_store() -> RecordingDescriptionStore:
    return RecordingDescriptionStore()


@pytest.fixture()
def manager(event_store: RecordingDescriptionStore) -> TeamManager:
    # No actor system is ever reached: the pass-through touches no runtime.
    return TeamManager(actor_system=MagicMock(), event_store=event_store)


class TestUpdateDescription:
    def test_a_user_write_returns_the_updated_process(
        self, manager: TeamManager, event_store: RecordingDescriptionStore
    ) -> None:
        saved = make_process()
        event_store.save_team(saved)

        result = manager.update_description(
            saved.team_id, "Quarterly review", DescriptionOrigin.USER
        )

        assert result is not None
        assert result.team_description == "Quarterly review"
        assert result.description_origin is DescriptionOrigin.USER
        assert event_store.load_team(saved.team_id) == result

    def test_an_auto_write_against_a_user_owned_record_returns_it_unchanged(
        self, manager: TeamManager, event_store: RecordingDescriptionStore
    ) -> None:
        saved = make_process()
        event_store.save_team(saved)
        owned = manager.update_description(saved.team_id, "Mine", DescriptionOrigin.USER)

        result = manager.update_description(saved.team_id, "Generated", DescriptionOrigin.AUTO)

        assert result == owned
        assert result is not None
        assert result.team_description == "Mine"
        assert result.description_origin is DescriptionOrigin.USER

    def test_an_unknown_team_returns_none(self, manager: TeamManager) -> None:
        assert manager.update_description(uuid.uuid4(), "Anything", DescriptionOrigin.USER) is None

    @pytest.mark.parametrize("origin", list(DescriptionOrigin), ids=lambda o: o.value)
    @pytest.mark.parametrize("description", ["A description", None])
    def test_the_store_is_called_exactly_once_with_the_arguments_handed_in(
        self,
        manager: TeamManager,
        event_store: RecordingDescriptionStore,
        origin: DescriptionOrigin,
        description: str | None,
    ) -> None:
        """Pass-through: one store write, no load first, no decision of its own."""
        saved = make_process()
        event_store.save_team(saved)
        event_store.load_team_calls = 0

        manager.update_description(saved.team_id, description, origin)

        assert event_store.description_calls == [(saved.team_id, description, origin)]
        assert event_store.load_team_calls == 0
