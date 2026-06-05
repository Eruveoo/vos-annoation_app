import React, { useCallback, useEffect, useMemo, useState } from "react";
import { getBehavior } from "./api.js";
import {
  BEHAVIOR_DIMENSIONS,
  BEHAVIOR_DIMENSION_TITLES,
  labelNameFi,
} from "./behaviorLabels.js";
import { colorForLabelId, isNeutralLabelId } from "./behaviorTimelineColors.js";

const ROW_HEIGHT = 16;
const COW_GAP = 10;
const LABEL_COL_WIDTH = 148;

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
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: "6px 14px",
                }}
              >
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

function TimelineRow({ segments, cowId, dimension, maxFrame, totalFrames, onSeekFrame }) {
  const cowSegs = segmentsForCowRow(segments, cowId);
  const spans = cowSegs
    .map((seg) => segmentSpan(seg, maxFrame))
    .filter(Boolean);

  const handleClick = (e) => {
    if (!onSeekFrame || totalFrames < 1) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const frame = Math.round(ratio * maxFrame);
    onSeekFrame(frame);
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
        cursor: onSeekFrame ? "pointer" : "default",
      }}
    >
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
            }}
          />
        );
      })}
    </div>
  );
}

/**
 * Per-cow timeline: 3 rows (Label 1–3) aligned to golden preview length.
 */
export default function BehaviorTimeline({
  runId,
  maxFrame = 0,
  currentFrame = 0,
  onSeekFrame = null,
  refreshToken = 0,
}) {
  const [behaviorData, setBehaviorData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

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

  const timelineHeight =
    cowIds.length * (ROW_HEIGHT * 3 + COW_GAP) - COW_GAP;

  return (
    <div
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
        Golden preview frames 0–{maxFrame}. Three rows per cow (Label 1–3). Click a row to jump the
        video to that frame.
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

      <div style={{ display: "flex", gap: 10, alignItems: "stretch" }}>
        <div style={{ width: LABEL_COL_WIDTH, flexShrink: 0 }} />
        <div style={{ flex: 1, minWidth: 0, position: "relative" }}>
          <div
            style={{
              position: "absolute",
              left: `${playheadPct}%`,
              top: 0,
              height: timelineHeight,
              width: 2,
              marginLeft: -1,
              backgroundColor: "#dc3545",
              zIndex: 3,
              pointerEvents: "none",
              boxShadow: "0 0 0 1px rgba(255,255,255,0.8)",
            }}
          />
          <div style={{ display: "flex", flexDirection: "column", gap: COW_GAP }}>
            {cowIds.map((cowId) => (
              <div key={cowId}>
                <div
                  style={{
                    fontSize: 12,
                    fontWeight: 700,
                    color: "#212529",
                    marginBottom: 4,
                  }}
                >
                  Cow {cowId}
                </div>
                {BEHAVIOR_DIMENSIONS.map((dim) => (
                  <div
                    key={dim}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      marginBottom: dim === "label3" ? 0 : 3,
                    }}
                  >
                    <div
                      style={{
                        width: LABEL_COL_WIDTH,
                        flexShrink: 0,
                        fontSize: 10,
                        color: "#6c757d",
                        lineHeight: 1.2,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                      title={BEHAVIOR_DIMENSION_TITLES[dim]}
                    >
                      {DIM_ROW_LABEL[dim]}
                    </div>
                    <TimelineRow
                      segments={dimensions[dim]?.segments}
                      cowId={cowId}
                      dimension={dim}
                      maxFrame={maxFrame}
                      totalFrames={totalFrames}
                      onSeekFrame={onSeekFrame}
                    />
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: 8,
          marginLeft: LABEL_COL_WIDTH + 10,
          fontSize: 10,
          color: "#868e96",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        <span>0</span>
        <span style={{ color: "#dc3545", fontWeight: 600 }}>▲ frame {currentFrame}</span>
        <span>{maxFrame}</span>
      </div>
    </div>
  );
}
