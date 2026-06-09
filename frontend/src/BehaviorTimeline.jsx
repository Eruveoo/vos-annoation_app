import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { deleteBehaviorLabel, getBehavior, setBehaviorLabel } from "./api.js";
import {
  BEHAVIOR_DIMENSIONS,
  BEHAVIOR_DIMENSION_TITLES,
  BEHAVIOR_LABELS_BY_DIMENSION,
  labelNameFi,
} from "./behaviorLabels.js";
import { colorForLabelId, isNeutralLabelId } from "./behaviorTimelineColors.js";

const ROW_HEIGHT = 16;
const COW_GAP = 10;
const LABEL_COL_WIDTH = 148;
const COL_GAP = 8;
const COW_HEADER_HEIGHT = 18;
const MARKER_HIT_WIDTH = 10;

function segmentSpan(seg, maxFrame) {
  const start = Math.max(0, Number(seg.start_frame) || 0);
  let end = seg.end_frame;
  if (end === null || end === undefined) {
    end = maxFrame;
  } else {
    end = Math.min(maxFrame, Number(end));
  }
  if (end < start) return null;
  return { start, end, labelId: seg.label_id };
}

function segmentsForCowRow(segments, cowId) {
  return (segments || [])
    .filter((s) => Number(s.cow_id) === Number(cowId))
    .map((s) => ({ ...s, cow_id: Number(s.cow_id) }))
    .sort((a, b) => a.start_frame - b.start_frame);
}

function getLabelAtFrame(segments, cowId, frame) {
  let best = null;
  for (const seg of segments || []) {
    if (Number(seg.cow_id) !== Number(cowId)) continue;
    const start = Number(seg.start_frame);
    const end = seg.end_frame;
    if (frame < start) continue;
    if (end !== null && end !== undefined && frame > Number(end)) continue;
    if (!best || start >= Number(best.start_frame)) best = seg;
  }
  return best?.label_id ?? null;
}

function hasChangeAtFrame(segments, cowId, frame) {
  return (segments || []).some(
    (s) => Number(s.cow_id) === Number(cowId) && Number(s.start_frame) === frame
  );
}

function catalogForDimension(dimensions, dimension) {
  const fromApi = dimensions[dimension]?.labels;
  if (fromApi?.length) {
    return fromApi.map((l) => ({
      id: l.id,
      nameFi: l.name_fi || l.nameFi || l.id,
      descriptionFi: l.description_fi || l.descriptionFi || "",
    }));
  }
  return BEHAVIOR_LABELS_BY_DIMENSION[dimension] || [];
}

function clickPosition(e, maxFrame) {
  const rect = e.currentTarget.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  return {
    frame: Math.round(ratio * maxFrame),
    xPct: ratio * 100,
  };
}

function cowBlockHeight() {
  return COW_HEADER_HEIGHT + 4 + ROW_HEIGHT * 3 + 3 * 2;
}

function rowTopPx(cowIds, cowId, dimension) {
  const cowIdx = cowIds.indexOf(cowId);
  if (cowIdx < 0) return 0;
  let top = cowIdx * (cowBlockHeight() + COW_GAP);
  top += COW_HEADER_HEIGHT + 4;
  const dimIdx = BEHAVIOR_DIMENSIONS.indexOf(dimension);
  if (dimIdx > 0) top += dimIdx * (ROW_HEIGHT + 3);
  return top;
}

function TimelineLegend({ usedByDimension }) {
  return (
    <div
      style={{
        marginBottom: 16,
        padding: 14,
        backgroundColor: "#fff",
        borderRadius: 10,
        border: "1px solid #e9ecef",
      }}
    >
      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10, color: "#212529" }}>
        Color legend
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {BEHAVIOR_DIMENSIONS.map((dim) => {
          const used = usedByDimension[dim] || [];
          if (used.length === 0) return null;
          return (
            <div key={dim}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#495057", marginBottom: 6 }}>
                {BEHAVIOR_DIMENSION_TITLES[dim]}
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 14px" }}>
                {used.map(({ id, nameFi }) => (
                  <span
                    key={`${dim}-${id}`}
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 6,
                      fontSize: 11,
                      color: "#495057",
                    }}
                  >
                    <span
                      style={{
                        width: 12,
                        height: 12,
                        borderRadius: 3,
                        backgroundColor: colorForLabelId(dim, id),
                        border: "1px solid rgba(0,0,0,0.12)",
                        flexShrink: 0,
                      }}
                    />
                    {nameFi}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ChangeMarker({
  startFrame,
  totalFrames,
  dimension,
  labelId,
  isActive,
  disabled,
  onMarkerClick,
}) {
  const leftPct = (startFrame / totalFrames) * 100;

  return (
    <button
      type="button"
      title={`${labelNameFi(labelId, dimension)} from frame ${startFrame}`}
      disabled={disabled}
      onClick={(e) => {
        e.stopPropagation();
        onMarkerClick(startFrame, leftPct);
      }}
      style={{
        position: "absolute",
        left: `${leftPct}%`,
        top: 0,
        bottom: 0,
        width: MARKER_HIT_WIDTH,
        marginLeft: -MARKER_HIT_WIDTH / 2,
        padding: 0,
        border: "none",
        background: "transparent",
        cursor: disabled ? "default" : "pointer",
        zIndex: 2,
      }}
    >
      <span
        style={{
          display: "block",
          width: 2,
          height: "100%",
          margin: "0 auto",
          backgroundColor: isActive ? "#212529" : "rgba(33,37,41,0.55)",
          borderRadius: 1,
          boxShadow: isActive ? "0 0 0 1px #fff" : "none",
        }}
      />
    </button>
  );
}

function TimelineRow({
  segments,
  cowId,
  dimension,
  maxFrame,
  totalFrames,
  anchor,
  disabled,
  onRowClick,
  onMarkerClick,
}) {
  const cowSegs = segmentsForCowRow(segments, cowId);
  const spans = cowSegs.map((seg) => segmentSpan(seg, maxFrame)).filter(Boolean);
  const markers = cowSegs.filter((s) => Number(s.start_frame) > 0);

  const handleClick = (e) => {
    if (disabled || maxFrame < 0) return;
    const { frame, xPct } = clickPosition(e, maxFrame);
    onRowClick?.(cowId, dimension, frame, xPct);
  };

  return (
    <div
      role="presentation"
      onClick={handleClick}
      style={{
        position: "relative",
        height: ROW_HEIGHT,
        flex: 1,
        minWidth: 0,
        backgroundColor: "#f1f3f5",
        borderRadius: 4,
        overflow: "hidden",
        cursor: disabled ? "default" : "pointer",
        isolation: "isolate",
      }}
    >
      <div style={{ position: "absolute", inset: 0, borderRadius: 4, pointerEvents: "none" }}>
        {spans.map((span, i) => {
          const leftPct = (span.start / totalFrames) * 100;
          const widthPct = ((span.end - span.start + 1) / totalFrames) * 100;
          const title = `${labelNameFi(span.labelId, dimension)} · frames ${span.start}–${span.end}`;
          return (
            <div
              key={`${span.start}-${span.labelId}-${i}`}
              title={title}
              style={{
                position: "absolute",
                left: `${leftPct}%`,
                width: `${Math.max(widthPct, 0.4)}%`,
                top: 0,
                bottom: 0,
                backgroundColor: colorForLabelId(dimension, span.labelId),
                borderRight: "1px solid rgba(255,255,255,0.5)",
                boxSizing: "border-box",
                pointerEvents: "none",
              }}
            />
          );
        })}
      </div>

      {markers.map((seg) => {
        const startFrame = Number(seg.start_frame);
        const isActive =
          anchor?.mode === "existing" &&
          anchor?.cowId === cowId &&
          anchor?.dimension === dimension &&
          anchor?.frame === startFrame;
        return (
          <ChangeMarker
            key={`marker-${startFrame}`}
            startFrame={startFrame}
            totalFrames={totalFrames}
            dimension={dimension}
            labelId={seg.label_id}
            isActive={isActive}
            disabled={disabled}
            onMarkerClick={(frame, xPct) =>
              onMarkerClick(cowId, dimension, frame, xPct)
            }
          />
        );
      })}
    </div>
  );
}

function TimelinePopover({
  anchor,
  popoverStep,
  cowIds,
  dimensions,
  disabled,
  busy,
  popoverError,
  onOpenMenu,
  onApplyLabel,
  onDeleteChange,
  onClose,
}) {
  if (!anchor) return null;

  const { cowId, dimension, frame, xPct, mode } = anchor;
  const top = rowTopPx(cowIds, cowId, dimension);
  const segs = dimensions[dimension]?.segments || [];
  const catalog = catalogForDimension(dimensions, dimension);
  const currentLabel = getLabelAtFrame(segs, cowId, frame);
  const dimTitle = dimensions[dimension]?.title_fi || BEHAVIOR_DIMENSION_TITLES[dimension];

  const stop = (e) => e.stopPropagation();

  const buttonStyle = {
    padding: "5px 10px",
    fontSize: 11,
    fontWeight: 600,
    border: "none",
    borderRadius: 6,
    cursor: disabled || busy ? "not-allowed" : "pointer",
    boxShadow: "0 2px 8px rgba(0,0,0,0.18)",
    whiteSpace: "nowrap",
  };

  return (
    <div
      onMouseDown={stop}
      onClick={stop}
      style={{
        position: "absolute",
        left: `${xPct}%`,
        top,
        transform: "translate(-50%, calc(-100% - 6px))",
        zIndex: 30,
        pointerEvents: "auto",
        display: "flex",
        flexDirection: "column",
        gap: 4,
        alignItems: "center",
      }}
    >
      {popoverStep === "button" && mode === "new" && (
        <button
          type="button"
          disabled={disabled || busy}
          onClick={onOpenMenu}
          style={{ ...buttonStyle, backgroundColor: "#0d6efd", color: "#fff" }}
        >
          Insert label
        </button>
      )}

      {popoverStep === "button" && mode === "existing" && (
        <>
          <button
            type="button"
            disabled={disabled || busy}
            onClick={onOpenMenu}
            style={{ ...buttonStyle, backgroundColor: "#0d6efd", color: "#fff" }}
          >
            Change label
          </button>
          <button
            type="button"
            disabled={disabled || busy}
            onClick={onDeleteChange}
            style={{ ...buttonStyle, backgroundColor: "#dc3545", color: "#fff" }}
          >
            Delete label
          </button>
        </>
      )}

      {popoverStep === "menu" && (
        <div
          style={{
            backgroundColor: "#fff",
            borderRadius: 8,
            border: "1px solid #dee2e6",
            boxShadow: "0 4px 16px rgba(0,0,0,0.15)",
            minWidth: 200,
            maxWidth: 280,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              padding: "8px 10px",
              borderBottom: "1px solid #e9ecef",
              backgroundColor: "#f8f9fa",
            }}
          >
            <div style={{ fontSize: 11, fontWeight: 700, color: "#212529" }}>
              Cow {cowId} · frame {frame}
            </div>
            <div style={{ fontSize: 10, color: "#0d6efd", marginTop: 2, fontWeight: 600 }}>
              {dimTitle}
            </div>
            {currentLabel && (
              <div style={{ fontSize: 10, color: "#6c757d", marginTop: 2 }}>
                Now: {labelNameFi(currentLabel, dimension)}
              </div>
            )}
          </div>

          {popoverError && (
            <div
              style={{
                padding: "6px 10px",
                fontSize: 10,
                color: "#721c24",
                backgroundColor: "#f8d7da",
              }}
            >
              {popoverError}
            </div>
          )}

          <div style={{ maxHeight: 220, overflowY: "auto" }}>
            {catalog.map((l) => (
              <button
                key={l.id}
                type="button"
                disabled={disabled || busy}
                title={l.descriptionFi}
                onClick={() => onApplyLabel(l.id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  width: "100%",
                  padding: "7px 10px",
                  border: "none",
                  borderBottom: "1px solid #f1f3f5",
                  backgroundColor: currentLabel === l.id ? "#e7f1ff" : "#fff",
                  cursor: disabled || busy ? "not-allowed" : "pointer",
                  fontSize: 12,
                  textAlign: "left",
                  color: "#212529",
                }}
              >
                <span
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: 2,
                    backgroundColor: colorForLabelId(dimension, l.id),
                    border: "1px solid rgba(0,0,0,0.1)",
                    flexShrink: 0,
                  }}
                />
                {l.nameFi}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={onClose}
            style={{
              width: "100%",
              padding: "6px 10px",
              border: "none",
              borderTop: "1px solid #e9ecef",
              backgroundColor: "#f8f9fa",
              color: "#6c757d",
              fontSize: 10,
              cursor: "pointer",
            }}
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * Per-cow timeline: 3 rows (Label 1–3). Click empty area to insert, click marker to edit/delete.
 */
export default function BehaviorTimeline({
  runId,
  maxFrame = 0,
  currentFrame = 0,
  onSeekFrame = null,
  onLabelChanged = null,
  refreshToken = 0,
  disabled = false,
}) {
  const [behaviorData, setBehaviorData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedCowId, setSelectedCowId] = useState(null);
  const [selectedDimension, setSelectedDimension] = useState(null);
  const [anchor, setAnchor] = useState(null);
  const [popoverStep, setPopoverStep] = useState(null);
  const [busy, setBusy] = useState(false);
  const [popoverError, setPopoverError] = useState("");
  const tracksRef = useRef(null);
  const timelineRef = useRef(null);

  const load = useCallback(async () => {
    if (!runId) return;
    setLoading(true);
    setError("");
    try {
      const data = await getBehavior(runId);
      setBehaviorData(data);
    } catch (e) {
      setError(e.message);
      setBehaviorData(null);
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    load();
  }, [load, refreshToken]);

  const closePopover = useCallback(() => {
    setAnchor(null);
    setPopoverStep(null);
    setPopoverError("");
  }, []);

  useEffect(() => {
    if (!anchor) return undefined;
    const handleOutside = (e) => {
      if (timelineRef.current?.contains(e.target)) return;
      closePopover();
    };
    document.addEventListener("mousedown", handleOutside);
    return () => document.removeEventListener("mousedown", handleOutside);
  }, [anchor, closePopover]);

  const dimensions = behaviorData?.dimensions || {};
  const cowIds = useMemo(() => {
    const ids = new Set();
    BEHAVIOR_DIMENSIONS.forEach((dim) => {
      (dimensions[dim]?.cow_ids || []).forEach((id) => ids.add(Number(id)));
    });
    return [...ids].sort((a, b) => a - b);
  }, [dimensions]);

  const usedByDimension = useMemo(() => {
    const out = {};
    BEHAVIOR_DIMENSIONS.forEach((dim) => {
      const seen = new Map();
      (dimensions[dim]?.segments || []).forEach((seg) => {
        const id = seg.label_id;
        if (id && !isNeutralLabelId(id)) {
          seen.set(id, labelNameFi(id, dim));
        }
      });
      out[dim] = [...seen.entries()].map(([id, nameFi]) => ({ id, nameFi }));
    });
    return out;
  }, [dimensions]);

  const DIM_ROW_LABEL = {
    activity: "L1 Aktivisuus",
    label2: "L2 Toimija",
    label3: "L3 Vastaanottaja",
  };

  const totalFrames = Math.max(1, maxFrame + 1);
  const playheadPct = maxFrame > 0 ? (currentFrame / maxFrame) * 100 : 0;

  const openAnchor = (cowId, dimension, frame, xPct, mode) => {
    if (disabled || busy) return;
    setSelectedCowId(cowId);
    setSelectedDimension(dimension);
    setAnchor({ cowId, dimension, frame, xPct, mode });
    setPopoverStep("button");
    setPopoverError("");
    onSeekFrame?.(frame);
  };

  const handleRowClick = (cowId, dimension, frame, xPct) => {
    if (hasChangeAtFrame(dimensions[dimension]?.segments, cowId, frame) && frame > 0) {
      openAnchor(cowId, dimension, frame, (frame / totalFrames) * 100, "existing");
      return;
    }
    openAnchor(cowId, dimension, frame, xPct, "new");
  };

  const handleMarkerClick = (cowId, dimension, frame, xPct) => {
    openAnchor(cowId, dimension, frame, xPct, "existing");
  };

  const handleApplyLabel = async (labelId) => {
    if (!anchor || !labelId || disabled) return;
    const current = getLabelAtFrame(
      dimensions[anchor.dimension]?.segments,
      anchor.cowId,
      anchor.frame
    );
    if (current === labelId) {
      closePopover();
      return;
    }

    setBusy(true);
    setPopoverError("");
    try {
      const res = await setBehaviorLabel(
        runId,
        anchor.cowId,
        anchor.frame,
        labelId,
        anchor.dimension
      );
      await load();
      onLabelChanged?.(anchor.dimension, res.preview_in_sync);
      closePopover();
    } catch (e) {
      setPopoverError(e.message);
      setPopoverStep("menu");
    } finally {
      setBusy(false);
    }
  };

  const handleDeleteChange = async () => {
    if (!anchor || disabled) return;
    setBusy(true);
    setPopoverError("");
    try {
      const res = await deleteBehaviorLabel(
        runId,
        anchor.cowId,
        anchor.frame,
        anchor.dimension
      );
      await load();
      onLabelChanged?.(anchor.dimension, res.preview_in_sync);
      closePopover();
    } catch (e) {
      setPopoverError(e.message);
      setPopoverStep("button");
    } finally {
      setBusy(false);
    }
  };

  if (!runId || maxFrame === null || maxFrame === undefined) {
    return null;
  }

  if (loading && !behaviorData) {
    return (
      <p style={{ fontSize: 13, color: "#6c757d", marginTop: 16 }}>Loading behaviour timeline…</p>
    );
  }

  if (error) {
    return (
      <div
        style={{
          marginTop: 16,
          padding: 12,
          backgroundColor: "#f8d7da",
          borderRadius: 8,
          color: "#721c24",
          fontSize: 13,
        }}
      >
        {error}
      </div>
    );
  }

  if (!cowIds.length) {
    return (
      <p style={{ fontSize: 13, color: "#6c757d", marginTop: 16 }}>
        No behaviour data yet. Complete ID assignment first.
      </p>
    );
  }

  return (
    <div
      ref={timelineRef}
      style={{
        marginTop: 20,
        marginBottom: 8,
        padding: 16,
        backgroundColor: "#f8f9fa",
        borderRadius: 12,
        border: "1px solid #e9ecef",
      }}
    >
      <h3 style={{ marginTop: 0, marginBottom: 6, fontSize: 16 }}>Behaviour timeline</h3>
      <p style={{ fontSize: 12, color: "#6c757d", marginTop: 0, marginBottom: 14, lineHeight: 1.5 }}>
        Click any row (L1–L3) to insert a label. Vertical ticks mark existing changes — click a tick
        to change or delete. Works independently per label dimension.
      </p>

      <TimelineLegend usedByDimension={usedByDimension} />
      <div style={{ fontSize: 11, color: "#868e96", marginBottom: 10 }}>
        <span
          style={{
            display: "inline-block",
            width: 12,
            height: 12,
            borderRadius: 3,
            backgroundColor: "#e9ecef",
            border: "1px solid rgba(0,0,0,0.12)",
            verticalAlign: "middle",
            marginRight: 6,
          }}
        />
        Grey = Ei valittu / Ei näy / Ei näkyvissä
      </div>

      <div style={{ display: "flex", gap: COL_GAP, alignItems: "flex-start" }}>
        <div style={{ width: LABEL_COL_WIDTH, flexShrink: 0 }}>
          {cowIds.map((cowId) => (
            <div key={cowId} style={{ marginBottom: COW_GAP }}>
              <div
                style={{
                  height: COW_HEADER_HEIGHT,
                  marginBottom: 4,
                  fontSize: 12,
                  fontWeight: 700,
                  color: selectedCowId === cowId ? "#0d6efd" : "#212529",
                  lineHeight: 1.2,
                  display: "flex",
                  alignItems: "flex-end",
                }}
              >
                Cow {cowId}
              </div>
              {BEHAVIOR_DIMENSIONS.map((dim, dimIdx) => (
                <div
                  key={dim}
                  style={{
                    height: ROW_HEIGHT,
                    marginBottom: dimIdx < BEHAVIOR_DIMENSIONS.length - 1 ? 3 : 0,
                    fontSize: 10,
                    color:
                      selectedCowId === cowId && selectedDimension === dim ? "#0d6efd" : "#6c757d",
                    fontWeight:
                      selectedCowId === cowId && selectedDimension === dim ? 700 : 400,
                    lineHeight: `${ROW_HEIGHT}px`,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                  title={BEHAVIOR_DIMENSION_TITLES[dim]}
                >
                  {DIM_ROW_LABEL[dim]}
                </div>
              ))}
            </div>
          ))}
        </div>

        <div ref={tracksRef} style={{ flex: 1, minWidth: 0, position: "relative" }}>
          <TimelinePopover
            anchor={anchor}
            popoverStep={popoverStep}
            cowIds={cowIds}
            dimensions={dimensions}
            disabled={disabled}
            busy={busy}
            popoverError={popoverError}
            onOpenMenu={() => setPopoverStep("menu")}
            onApplyLabel={handleApplyLabel}
            onDeleteChange={handleDeleteChange}
            onClose={closePopover}
          />

          <div style={{ display: "flex", flexDirection: "column", gap: COW_GAP }}>
            {cowIds.map((cowId) => (
              <div key={cowId} style={{ position: "relative" }}>
                <div style={{ height: COW_HEADER_HEIGHT, marginBottom: 4 }} />
                {BEHAVIOR_DIMENSIONS.map((dim, dimIdx) => (
                  <div
                    key={dim}
                    style={{
                      position: "relative",
                      zIndex: dimIdx + 1,
                      marginBottom: dimIdx < BEHAVIOR_DIMENSIONS.length - 1 ? 3 : 0,
                    }}
                  >
                    {selectedCowId === cowId && selectedDimension === dim && (
                      <div
                        style={{
                          position: "absolute",
                          left: `${playheadPct}%`,
                          top: 0,
                          height: ROW_HEIGHT,
                          width: 2,
                          marginLeft: -1,
                          backgroundColor: "#dc3545",
                          zIndex: 6,
                          pointerEvents: "none",
                          boxShadow: "0 0 0 1px rgba(255,255,255,0.8)",
                        }}
                      />
                    )}
                    <TimelineRow
                      segments={dimensions[dim]?.segments}
                      cowId={cowId}
                      dimension={dim}
                      maxFrame={maxFrame}
                      totalFrames={totalFrames}
                      anchor={anchor}
                      disabled={disabled || busy}
                      onRowClick={handleRowClick}
                      onMarkerClick={handleMarkerClick}
                    />
                  </div>
                ))}
              </div>
            ))}
          </div>

          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginTop: 8,
              fontSize: 10,
              color: "#868e96",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            <span>0</span>
            <span style={{ color: "#dc3545", fontWeight: 600 }}>
              {selectedCowId !== null && selectedDimension
                ? `▲ cow ${selectedCowId} · ${DIM_ROW_LABEL[selectedDimension]} · frame ${currentFrame}`
                : `▲ frame ${currentFrame}`}
            </span>
            <span>{maxFrame}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
