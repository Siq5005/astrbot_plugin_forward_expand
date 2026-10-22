"""Expand merged-forward messages before they reach the model.

Two upstream behaviours combine to swallow an incoming merged forward:

1. ``astr_message_event.get_message_str()`` renders a ``Forward`` component as
   the literal string ``[转发消息]`` without ever fetching its content.
2. ``agent_sub_stages/internal.py`` skips the LLM request entirely when
   ``event.message_str`` is blank and the message carries no image/file/record/
   video/reply component. A forward is none of those, so a forward-only message
   produces no reply at all.

An ``on_llm_request`` hook cannot fix this, because that hook only runs *after*
the skip decision. This plugin therefore registers a high-priority message
handler: it resolves every ``Forward`` component through the OneBot
``get_forward_msg`` action, writes the transcript back into
``event.message_str``, and appends the forwarded images to the message chain so
vision models actually see them instead of a bare ``[图片]`` marker.
"""

from __future__ import annotations

import sys
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Forward, Image
from astrbot.api.star import Context, Star, register

# Bounds so a huge forward cannot eat the whole context window.
MAX_FORWARDS = 3
MAX_NODES = 80
MAX_CHARS = 3000
MAX_NESTED_DEPTH = 2
MAX_IMAGES = 10


@register(
    "astrbot_plugin_forward_expand",
    "Coe",
    "把直接发送的合并转发消息展开成文本喂给模型",
    "1.1.0",
)
class ForwardExpandPlugin(Star):
    """Resolve incoming merged-forward messages into readable text."""

    def __init__(self, context: Context) -> None:
        """Store the plugin context.

        Args:
            context: AstrBot plugin context.
        """
        super().__init__(context)

    async def _call_get_forward_msg(
        self, event: AstrMessageEvent, forward_id: str
    ) -> dict[str, Any] | None:
        """Fetch one merged-forward payload through the OneBot API.

        NapCat and other OneBot implementations disagree on whether the parameter
        is called ``id`` or ``message_id``, and some want an integer, so both are
        tried before giving up.

        Args:
            event: Incoming message event carrying the platform bot client.
            forward_id: Forward message identifier taken from the component.

        Returns:
            The unwrapped payload dict, or None when every attempt failed.
        """
        api = getattr(getattr(event, "bot", None), "api", None)
        if api is None or not hasattr(api, "call_action"):
            logger.warning("[forward-expand] 当前平台不支持 call_action，跳过展开")
            return None

        attempts: list[dict[str, Any]] = [
            {"id": forward_id},
            {"message_id": forward_id},
        ]
        if forward_id.isdigit():
            attempts.extend([{"id": int(forward_id)}, {"message_id": int(forward_id)}])

        for params in attempts:
            try:
                ret = await api.call_action("get_forward_msg", **params)
            except Exception as exc:  # noqa: BLE001 - any API error means try next
                logger.debug(f"[forward-expand] get_forward_msg {params} 失败: {exc}")
                continue
            if not isinstance(ret, dict):
                continue
            logger.debug(f"[forward-expand] get_forward_msg 命中参数 {params}")
            # WebSocket mode returns the payload directly; HTTP mode nests it.
            data = ret.get("data")
            return data if isinstance(data, dict) else ret
        logger.warning(f"[forward-expand] get_forward_msg 全部失败，id={forward_id}")
        return None

    def _render_segments(
        self,
        segments: Any,
        depth: int = 0,
        budget: list[int] | None = None,
        image_sources: list[str] | None = None,
    ) -> str:
        """Flatten an OneBot message segment list into plain text.

        Args:
            segments: Segment list from a forward node.
            depth: Current node nesting depth, used to bound recursion.
            budget: Shared mutable remaining-node counter.
            image_sources: Shared list collecting image URLs for later attachment.

        Returns:
            Best-effort one-line rendering of the segments.
        """
        if not isinstance(segments, list):
            return ""
        if budget is None:
            budget = [MAX_NODES]
        if image_sources is None:
            image_sources = []

        parts: list[str] = []
        for seg in segments:
            if budget[0] <= 0:
                parts.append("…")
                break
            if not isinstance(seg, dict):
                continue
            kind = seg.get("type")
            data = seg.get("data") if isinstance(seg.get("data"), dict) else {}

            if kind == "text":
                text = data.get("text")
                if text:
                    parts.append(str(text))
            elif kind == "image":
                # Keep a numbered marker so the model can line up the text with
                # the attachments appended to the message chain.
                source = str(data.get("url") or data.get("file") or "").strip()
                if source and len(image_sources) < MAX_IMAGES:
                    image_sources.append(source)
                    parts.append(f"[图片{len(image_sources)}]")
                else:
                    parts.append("[图片]")
            elif kind == "face":
                parts.append("[表情]")
            elif kind in ("at", "mention"):
                parts.append(f"@{data.get('qq') or data.get('name') or ''}")
            elif kind == "node" and depth < MAX_NESTED_DEPTH:
                budget[0] -= 1
                sender = (data.get("sender") or {}).get("nickname") or data.get(
                    "nickname"
                )
                body = self._render_segments(
                    data.get("content"), depth + 1, budget, image_sources
                )
                if body:
                    parts.append(f"{sender}: {body}" if sender else body)
            elif kind == "forward":
                parts.append("[嵌套转发]")
        return " ".join(p for p in parts if p).strip()

    def _render_forward_payload(
        self, payload: dict[str, Any], image_sources: list[str]
    ) -> str:
        """Turn a ``get_forward_msg`` payload into a transcript.

        Args:
            payload: Unwrapped payload containing the forward nodes.
            image_sources: Shared list collecting image URLs found in the nodes.

        Returns:
            Newline-joined ``nickname: text`` lines, possibly truncated.
        """
        nodes = (
            payload.get("messages")
            or payload.get("message")
            or payload.get("nodes")
            or payload.get("nodeList")
        )
        if not isinstance(nodes, list) or not nodes:
            return ""

        lines: list[str] = []
        budget = [MAX_NODES]
        for node in nodes:
            if budget[0] <= 0:
                lines.append("…（内容过长已截断）")
                break
            if not isinstance(node, dict):
                continue
            budget[0] -= 1
            # Two known shapes: {"type": "node", "data": {...}} and a bare node.
            data = node.get("data") if node.get("type") == "node" else node
            if not isinstance(data, dict):
                continue
            sender = (
                (data.get("sender") or {}).get("nickname")
                or data.get("nickname")
                or data.get("name")
                or "?"
            )
            body = self._render_segments(
                data.get("content") or data.get("message"),
                1,
                budget,
                image_sources,
            )
            if body:
                lines.append(f"{sender}: {body}")

        text = "\n".join(lines)
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "\n…（已截断）"
        return text

    @filter.event_message_type(filter.EventMessageType.ALL, priority=sys.maxsize - 1)
    async def expand_forward(self, event: AstrMessageEvent) -> None:
        """Write merged-forward transcripts and images into the event.

        The event is deliberately left running: with ``message_str`` no longer
        blank, the normal agent stage builds a reply instead of discarding the
        message. Forwarded images are appended to the message chain so the
        agent's attachment loop feeds them to the model as real images.

        Args:
            event: Incoming message event.
        """
        forward_ids = [
            str(seg.id).strip()
            for seg in event.get_messages()
            if isinstance(seg, Forward) and str(getattr(seg, "id", "")).strip()
        ]
        if not forward_ids:
            return

        image_sources: list[str] = []
        try:
            blocks: list[str] = []
            for forward_id in forward_ids[:MAX_FORWARDS]:
                payload = await self._call_get_forward_msg(event, forward_id)
                if not payload:
                    continue
                transcript = self._render_forward_payload(payload, image_sources)
                if transcript:
                    blocks.append(transcript)
        except Exception as exc:  # noqa: BLE001 - never break the chat pipeline
            logger.error(f"[forward-expand] 展开失败: {exc}")
            blocks = []

        original = event.message_str.strip()
        if blocks:
            body = "\n\n".join(blocks)
            event.message_str = (
                f"{original}\n\n[合并转发内容]\n{body}"
                if original
                else f"[合并转发内容]\n{body}"
            )
            # Appending to the chain lets the agent's attachment loop download
            # these and pass them to the model as actual images. ``url`` is set
            # alongside ``file`` so other plugins that read ``.url`` (e.g. the
            # GsCore adapter) recognise the segment instead of warning about a
            # missing path.
            for source in image_sources:
                event.message_obj.message.append(Image(file=source, url=source))
            logger.info(
                f"[forward-expand] 展开 {len(blocks)} 个合并转发，"
                f"共 {len(body)} 字，附带 {len(image_sources)} 张图"
            )
            return

        # Fall back to a marker so the message is still non-empty and the model
        # can answer honestly instead of the pipeline discarding it in silence.
        marker = "[用户发来一条合并转发消息，但内容读取失败]"
        event.message_str = f"{original}\n\n{marker}" if original else marker
        logger.warning(f"[forward-expand] 未取到内容，id={forward_ids}")
