import React, { useCallback, useEffect, useState } from "react";
import { getBehavior, setBehaviorLabel } from "./api.js";
import {
  BEHAVIOR_DIMENSIONS,
  BEHAVIOR_DIMENSION_TITLES,
  BEHAVIOR_LABELS_BY_DIMENSION,
  labelNameFi,
} from "./behaviorLabels.js";

function parseLabelsAtFrame(raw) {
  const at = {};
  Object.entries(raw || {}).forEach(([k, v]) => {
    at[Number(k)] = v;
  });
  return at;
}

/**
 * Edit per-cow behaviour labels (3 dimensions) from the video playhead frame onward.
 */
export default function BehaviorEditor({
  runId,
  frame = 0,
  maxFrame = null,
  onLabelChanged = null,
  disabled = false,
}) {
  const [cowIds, setCowIds] = useState([]);
  const [labelsByDim, setLabelsByDim] = useState({
    activity: {},
    label2: {},
    label3: {},
  });
  const [busyKey, setBusyKey] = useState(null);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const loadAtFrame = useCallback(
    async (f) => {
      if (!runId) return;
      const data = await getBehavior(runId, f);
      const dims = data.dimensions || {};
      const firstCowIds =
        dims.activity?.cow_ids ||
        dims.label2?.cow_ids ||
        dims.label3?.cow_ids ||
        data.cow_ids ||
        [];
      setCowIds(firstCowIds);
      setLabelsByDim({
        activity: parseLabelsAtFrame(dims.activity?.labels_at_frame),
        label2: parseLabelsAtFrame(dims.label2?.labels_at_frame),
        label3: parseLabelsAtFrame(dims.label3?.labels_at_frame),
      });
    },
    [runId]
  );

  useEffect(() => {
    loadAtFrame(frame).catch((e) => setError(e.message));
  }, [runId, frame, loadAtFrame]);

  const handleLabelChange = async (cowId, dimension, labelId) => {
    const current = labelsByDim[dimension]?.[cowId];
    if (!labelId || current === labelId) return;
    const busy = `${dimension}-${cowId}`;
    setBusyKey(busy);
    setError("");
    setSuccess("");
    try {
      const res = await setBehaviorLabel(runId, cowId, frame, labelId, dimension);
      await loadAtFrame(frame);
      const rebuildHint =
        res.preview_in_sync === false ? " Rebuild golden preview to update video." : "";
      setSuccess(
        `Cow ${cowId} · ${BEHAVIOR_DIMENSION_TITLES[dimension]}: ${labelNameFi(labelId, dimension)} from frame ${frame}.${rebuildHint}`
      );
      onLabelChanged?.(dimension, res.preview_in_sync);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusyKey(null);
    }
  };

  if (!cowIds.length) {
    return (
      <p style={{ color: "#6c757d", fontSize: 14 }}>
        No cows with behaviour labels yet. Complete ID assignment with behaviour labels first.
      </p>
    );
  }

  return (
    <div
      style={{
        marginTop: 16,
        padding: 20,
        backgroundColor: "#f8f9fa",
        borderRadius: 12,
        border: "1px solid #e9ecef",
      }}
    >
      <h3 style={{ marginTop: 0, marginBottom: 8 }}>Behaviour labels</h3>
      <p style={{ fontSize: 13, color: "#6c757d", marginTop: 0, lineHeight: 1.5 }}>
        Scrub the golden video to the frame where behaviour changes. <strong>Label 1 (Aktivisuus)</strong> is
        required; Labels 2 and 3 default to &quot;Ei valittu&quot; when not applicable.
      </p>

      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
        <span style={{ fontSize: 14, fontWeight: 600 }}>From frame:</span>
        <span
          style={{
            padding: "8px 14px",
            borderRadius: 8,
            backgroundColor: "#e9ecef",
            fontSize: 15,
            fontWeight: 700,
            fontVariantNumeric: "tabular-nums",
            minWidth: 56,
            textAlign: "center",
          }}
        >
          {frame}
        </span>
        {maxFrame !== null && maxFrame !== undefined && (
          <span style={{ fontSize: 12, color: "#6c757d" }}>(golden: 0–{maxFrame})</span>
        )}
      </div>

      {error && (
        <div
          style={{
            padding: 10,
            marginBottom: 12,
            backgroundColor: "#f8d7da",
            borderRadius: 8,
            color: "#721c24",
            fontSize: 13,
          }}
        >
          {error}
        </div>
      )}
      {success && (
        <div
          style={{
            padding: 10,
            marginBottom: 12,
            backgroundColor: "#d4edda",
            borderRadius: 8,
            color: "#155724",
            fontSize: 13,
          }}
        >
          {success}
        </div>
      )}

      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, minWidth: 720 }}>
          <thead>
            <tr style={{ backgroundColor: "#e9ecef" }}>
              <th style={{ padding: 10, textAlign: "left", border: "1px solid #dee2e6", width: 70 }}>
                Cow ID
              </th>
              {BEHAVIOR_DIMENSIONS.map((dim) => (
                <th
                  key={dim}
                  style={{ padding: 10, textAlign: "left", border: "1px solid #dee2e6", minWidth: 200 }}
                >
                  {BEHAVIOR_DIMENSION_TITLES[dim]}
                  {dim !== "activity" && (
                    <div style={{ fontSize: 11, fontWeight: 400, color: "#6c757d" }}>optional</div>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cowIds.map((cowId) => (
              <tr key={cowId}>
                <td style={{ padding: 10, border: "1px solid #dee2e6", fontWeight: 600 }}>{cowId}</td>
                {BEHAVIOR_DIMENSIONS.map((dim) => {
                  const catalog = BEHAVIOR_LABELS_BY_DIMENSION[dim];
                  const value = labelsByDim[dim]?.[cowId] || "";
                  const busy = busyKey === `${dim}-${cowId}`;
                  return (
                    <td key={dim} style={{ padding: 8, border: "1px solid #dee2e6" }}>
                      <div style={{ fontSize: 11, color: "#6c757d", marginBottom: 4 }}>
                        Now: {labelNameFi(value || "—", dim)}
                      </div>
                      <select
                        value={value}
                        disabled={disabled || busy}
                        onChange={(e) => handleLabelChange(cowId, dim, e.target.value)}
                        style={{
                          width: "100%",
                          padding: "6px 8px",
                          borderRadius: 6,
                          border: "1px solid #ccc",
                          fontSize: 12,
                        }}
                      >
                        {!value && <option value="">—</option>}
                        {catalog.map((l) => (
                          <option key={l.id} value={l.id} title={l.descriptionFi}>
                            {l.nameFi}
                          </option>
                        ))}
                      </select>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
