from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem


class ClickableRectItem(QGraphicsRectItem):
    """Read-only rect with hover highlight + right/left/double click callbacks."""

    HOVER_PEN_COLOR = QColor(255, 255, 0)

    def __init__(
        self,
        rect: QRectF,
        idx: int,
        *,
        on_right_click: Optional[Callable[[int], None]] = None,
        on_double_click: Optional[Callable[[int], None]] = None,
        on_left_click: Optional[Callable[[int], None]] = None,
        parent: Optional[QGraphicsItem] = None,
    ):
        super().__init__(rect, parent)
        self.idx = idx
        self.on_right_click = on_right_click
        self.on_double_click = on_double_click
        self.on_left_click = on_left_click
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._base_pen: QPen = QPen(self.pen())
        self.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton
        )

    def setPen(self, pen):  # type: ignore[override]
        super().setPen(pen)
        self._base_pen = QPen(pen)

    def hoverEnterEvent(self, event):  # type: ignore[override]
        pen = QPen(self._base_pen)
        pen.setWidth(self._base_pen.width() + 2)
        pen.setColor(self.HOVER_PEN_COLOR)
        super().setPen(pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):  # type: ignore[override]
        super().setPen(self._base_pen)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):  # type: ignore[override]
        btn = event.button()
        if btn == Qt.MouseButton.RightButton and self.on_right_click is not None:
            self.on_right_click(self.idx)
            event.accept()
            return
        if btn == Qt.MouseButton.LeftButton and self.on_left_click is not None:
            self.on_left_click(self.idx)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self.on_double_click is not None:
            self.on_double_click(self.idx)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class EditableBBoxItem(QGraphicsRectItem):
    """A bbox rectangle that can be moved by dragging the interior and resized
    by dragging near a corner. Right-click invokes ``on_right_click`` (typical
    delete handler). When ``edit_enabled`` is ``False``, the item behaves
    like :class:`ClickableRectItem` (no move/resize).

    Coordinates are in scene space; ``self.rect()`` is updated directly and
    ``self.pos()`` is left at the origin throughout, which keeps the rest of
    the GUI free from coordinate conversions.
    """

    HOVER_PEN_COLOR = QColor(255, 255, 0)
    HANDLE_SIZE = 12  # scene-pixel hit radius for corners
    MIN_SIZE = 6

    def __init__(
        self,
        rect: QRectF,
        idx: int,
        *,
        on_right_click: Optional[Callable[[int], None]] = None,
        on_geometry_changed: Optional[
            Callable[[int, int, int, int, int], None]
        ] = None,
        edit_enabled: bool = False,
        parent: Optional[QGraphicsItem] = None,
    ):
        super().__init__(rect, parent)
        self.idx = idx
        self._on_right_click = on_right_click
        self._on_geometry_changed = on_geometry_changed
        self._edit_enabled = edit_enabled
        self._dragging: Optional[str] = None  # None | 'move' | 'tl' | 'tr' | 'bl' | 'br'
        self._press_scene: Optional[QPointF] = None
        self._press_rect: Optional[QRectF] = None

        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton
        )
        self._base_pen: QPen = QPen(self.pen())

    # ---- pen / hover ----

    def setPen(self, pen):  # type: ignore[override]
        super().setPen(pen)
        self._base_pen = QPen(pen)

    def setEditEnabled(self, enabled: bool) -> None:
        self._edit_enabled = enabled
        self._update_cursor(corner=None)

    def hoverMoveEvent(self, event):  # type: ignore[override]
        corner = self._hit_corner(event.scenePos())
        self._update_cursor(corner)
        super().hoverMoveEvent(event)

    def hoverEnterEvent(self, event):  # type: ignore[override]
        pen = QPen(self._base_pen)
        pen.setWidth(self._base_pen.width() + 2)
        pen.setColor(self.HOVER_PEN_COLOR)
        super().setPen(pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):  # type: ignore[override]
        super().setPen(self._base_pen)
        super().hoverLeaveEvent(event)

    def _update_cursor(self, corner: Optional[str]) -> None:
        if corner in ("tl", "br"):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif corner in ("tr", "bl"):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif self._edit_enabled:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ---- hit testing ----

    def _hit_corner(self, scene_pos: QPointF) -> Optional[str]:
        if not self._edit_enabled:
            return None
        r = self.rect()
        s = self.HANDLE_SIZE
        for name, corner in (
            ("tl", r.topLeft()),
            ("tr", r.topRight()),
            ("bl", r.bottomLeft()),
            ("br", r.bottomRight()),
        ):
            if (
                abs(scene_pos.x() - corner.x()) <= s
                and abs(scene_pos.y() - corner.y()) <= s
            ):
                return name
        return None

    # ---- mouse handlers ----

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton and self._on_right_click is not None:
            self._on_right_click(self.idx)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._edit_enabled:
            corner = self._hit_corner(event.scenePos())
            self._dragging = corner if corner else "move"
            self._press_scene = event.scenePos()
            self._press_rect = QRectF(self.rect())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._dragging is None or self._press_rect is None or self._press_scene is None:
            super().mouseMoveEvent(event)
            return
        delta = event.scenePos() - self._press_scene
        r = QRectF(self._press_rect)
        if self._dragging == "move":
            r.translate(delta)
        elif self._dragging == "tl":
            r.setTopLeft(self._press_rect.topLeft() + delta)
        elif self._dragging == "tr":
            r.setTopRight(self._press_rect.topRight() + delta)
        elif self._dragging == "bl":
            r.setBottomLeft(self._press_rect.bottomLeft() + delta)
        elif self._dragging == "br":
            r.setBottomRight(self._press_rect.bottomRight() + delta)
        if r.width() < self.MIN_SIZE or r.height() < self.MIN_SIZE:
            return
        self.setRect(r.normalized())
        event.accept()

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if self._dragging is not None:
            self._dragging = None
            self._press_rect = None
            self._press_scene = None
            if self._on_geometry_changed is not None:
                r = self.rect()
                self._on_geometry_changed(
                    self.idx,
                    int(round(r.x())),
                    int(round(r.y())),
                    int(round(r.width())),
                    int(round(r.height())),
                )
            event.accept()
            return
        super().mouseReleaseEvent(event)


_FONT_FAMILY_CACHE: dict[str, str] = {}


def _resolve_font_family(font_path: Optional[str]) -> Optional[str]:
    """Register a font file with QFontDatabase and return its family name.

    Cached per-path so repeated loads (one per draggable text item) are cheap.
    Returns ``None`` if loading fails or no path was given.
    """
    if not font_path:
        return None
    if font_path in _FONT_FAMILY_CACHE:
        return _FONT_FAMILY_CACHE[font_path]
    from PySide6.QtGui import QFontDatabase

    font_id = QFontDatabase.addApplicationFont(font_path)
    if font_id == -1:
        _FONT_FAMILY_CACHE[font_path] = ""
        return None
    families = QFontDatabase.applicationFontFamilies(font_id)
    family = families[0] if families else ""
    _FONT_FAMILY_CACHE[font_path] = family
    return family or None


class DraggableTextItem(QGraphicsItem):
    """A draggable text overlay used in the Translate tab to reposition a
    rendered dialogue without moving its bbox.

    Layout: the text is split on user newlines and each line is laid out
    horizontally centered around the item's local origin (0, 0), so that
    multi-line previews always appear center-aligned regardless of the
    stored alignment. The item position is set so that local (0, 0) lands
    on ``bbox_center + offset``. Stored ``rotation`` is applied via
    :meth:`QGraphicsItem.setRotation` around the same origin.

    On release we emit the new offset (delta from the bbox center) via
    ``on_offset_changed``.
    """

    def __init__(
        self,
        idx: int,
        bbox_center: QPointF,
        offset: QPointF,
        text: str,
        *,
        on_offset_changed: Callable[[int, int, int], None],
        font_size: int = 24,
        font_path: Optional[str] = None,
        rotation: int = 0,
        parent: Optional[QGraphicsItem] = None,
    ):
        super().__init__(parent)
        self.idx = idx
        self._bbox_center = bbox_center
        self._on_offset_changed = on_offset_changed
        self._font_size = font_size

        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QGraphicsSimpleTextItem

        family = _resolve_font_family(font_path)
        font = QFont(family) if family else QFont()
        # Match PIL's pixel-height interpretation so the preview matches
        # the actual rendered output.
        font.setPixelSize(max(1, int(font_size)))

        # Lay out one QGraphicsSimpleTextItem per line, all centered.
        self._line_items = []
        lines = text.split("\n") if text else [""]
        line_h = max(1, int(font_size * 1.15))
        total_h = line_h * len(lines)
        y = -total_h / 2.0
        for line in lines:
            item = QGraphicsSimpleTextItem(line, self)
            item.setFont(font)
            item.setBrush(QColor("black"))
            rect = item.boundingRect()
            item.setPos(-rect.width() / 2.0, y)
            self._line_items.append(item)
            y += line_h

        self.setPos(self._bbox_center + offset)
        if rotation:
            self.setRotation(-int(rotation))  # Qt is CW-positive; PIL is CCW-positive
        self.setTransformOriginPoint(0, 0)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

    def boundingRect(self) -> QRectF:
        if not self._line_items:
            return QRectF()
        rect = QRectF()
        for item in self._line_items:
            r = item.boundingRect().translated(item.pos())
            rect = rect.united(r) if not rect.isNull() else r
        return rect

    def paint(self, painter, option, widget=None):  # type: ignore[override]
        # Children paint themselves; nothing to draw here.
        return

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        super().mouseReleaseEvent(event)
        new_offset = self.scenePos() - self._bbox_center
        self._on_offset_changed(
            self.idx, int(round(new_offset.x())), int(round(new_offset.y()))
        )
