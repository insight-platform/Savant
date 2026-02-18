"""Meta Merge Python handler - merges objects from two ROI pipelines using export/import."""

from typing import Any, Optional

from savant_rs import register_handler
from savant_rs.logging import LogLevel, log
from savant_rs.match_query import MatchQuery, StringExpression
from savant_rs.utils.serialization import Message

from roi_constants import EXPECTED_INGRESS_COUNT, LEFT_ROI_LABEL, RIGHT_ROI_LABEL, ROI_NAMESPACE


class MergeHandler:
    """Merges objects from incoming frame into current state using export/import."""

    def __call__(
        self,
        ingress_name: str,
        topic: str,
        current_state: Any,
        incoming_state: Optional[Any],
    ) -> bool:
        """Merge incoming metadata into current state.

        Exports only trees under ROI objects (router.left_roi / router.right_roi)
        and imports them into the current frame.
        """
        if incoming_state is not None:
            # Merge objects from incoming frame into current frame
            current_frame = current_state.video_frame
            incoming_frame = incoming_state.video_frame

            # Export only trees under ROI objects (ROI + its children)
            roi_query = MatchQuery.and_(
                MatchQuery.namespace(StringExpression.eq(ROI_NAMESPACE)),
                MatchQuery.label(
                    StringExpression.one_of(LEFT_ROI_LABEL, RIGHT_ROI_LABEL)
                ),
            )
            trees = incoming_frame.export_complete_object_trees(
                roi_query, delete_exported=False
            )

            if trees:
                # Import object trees into current frame
                current_frame.import_object_trees(trees)
                log(
                    LogLevel.Debug,
                    "meta_merge",
                    f"Merged {len(trees)} object tree(s) from {ingress_name}",
                )

        # Track received ingresses
        received = current_state.state.get("received_ingresses", set())
        received.add(ingress_name)
        current_state.state["received_ingresses"] = received

        # Mark ready when we have received from all expected ingresses
        return len(received) >= EXPECTED_INGRESS_COUNT


class HeadExpiredHandler:
    """Handles expired frames - forward with partial merge."""

    def __call__(self, state: Any) -> Optional[Message]:
        """Forward the frame even if not all ingresses have arrived."""
        return state.video_frame.to_message()


class HeadReadyHandler:
    """Handles ready frames - forward the merged frame."""

    def __call__(self, state: Any) -> Optional[Message]:
        """Forward the merged frame."""
        return state.video_frame.to_message()


class LateArrivalHandler:
    """Handles late-arriving frames."""

    def __call__(self, state: Any) -> None:
        """Log late arrival."""
        log(LogLevel.Warn, "meta_merge", "Late frame arrival ignored")


class UnsupportedMessageHandler:
    """Handles unsupported messages."""

    def __call__(
        self,
        ingress_name: str,
        topic: str,
        message: Message,
        data: list[bytes],
    ) -> None:
        """Log unsupported message."""
        log(
            LogLevel.Debug,
            "meta_merge",
            f"Unsupported message from {ingress_name}: {type(message)}",
        )


def init(params: Any) -> bool:
    """Register all callback handlers."""
    register_handler("merge_handler", MergeHandler())
    register_handler("head_expired_handler", HeadExpiredHandler())
    register_handler("head_ready_handler", HeadReadyHandler())
    register_handler("late_arrival_handler", LateArrivalHandler())
    register_handler("unsupported_message_handler", UnsupportedMessageHandler())
    log(LogLevel.Info, "meta_merge", "Meta Merge handler initialized")
    return True
