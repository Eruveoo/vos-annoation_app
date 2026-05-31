import React, { useCallback, useEffect, useState } from "react";
import { getBehavior, setBehaviorLabel } from "./api.js";
import { BEHAVIOR_LABELS, labelNameFi } from "./behaviorLabels.js";

/**
 * Edit per-cow behaviour labels from a given frame onward (segment-based).
 */
export default function BehaviorEditor({
  runId,
  defaultFrame = 0,
  maxFrame = null,
  videoFrame = null,
  onLabelUpdated = null,
  rebuildingPreview = false,
}) {
  const [frame, setFrame] = useState(defaultFrame);
  const [cowIds, setCowIds] = useState([]);
  const [labelsAtFrame, setLabelsAtFrame] = useState({});
  const [busyCowId, setBusyCowId] = useState(null);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const loadAtFrame = useCallback(
    async (f) => {
      if (!runId) return;
      const data = await getBehavior(runId, f);
      setCowIds(data.cow_ids || []);
      const at = {};
      Object.entries(data.labels_at_frame || {}).forEach(([k, v]) => {
        at[Number(k)] = v;
      });
      setLabelsAtFrame(at);
    },
    [runId]
  );

  useEffect(() => {
    setFrame(defaultFrame);
  }, [defaultFrame]);

  useEffect(() => {
    loadAtFrame(frame).catch((e) => setError(e.message));
  }, [runId, frame, loadAtFrame]);

  const handleLabelChange = async (cowId, labelId) => {
    if (!labelId || labelsAtFrame[cowId] === labelId) return;
    setBusyCowId(cowId);
    setError("");
    setSuccess("");
    try {
      await setBehaviorLabel(runId, cowId, frame, labelId);
      await loadAtFrame(frame);
      setSuccess(
        `Cow ${cowId}: ${labelNameFi(labelId)} from frame ${frame}. Updating golden preview…`
      );
      onLabelUpdated?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusyCowId(null);
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
        Labels are drawn on the golden preview under each cow ID. Set the frame where the behaviour
        changes, then pick a new label for that cow.
      </p>

      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
        <label style={{ fontSize: 14, fontWeight: 600 }}>From frame:</label>
        <input
          type="number"
          min={0}
          max={maxFrame ?? undefined}
          value={frame}
          onChange={(e) => setFrame(Math.max(0, parseInt(e.target.value, 10) || 0))}
          style={{
            width: 100,
            padding: "8px 10px",
            border: "1px solid #dee2e6",
            borderRadius: 8,
            fontSize: 14,
          }}
        />
        {videoFrame !== null && videoFrame !== undefined && (
          <button
            type="button"
            onClick={() => setFrame(videoFrame)}
            style={{
              padding: "8px 12px",
              fontSize: 13,
              fontWeight: 600,
              border: "1px solid #0d6efd",
              borderRadius: 8,
              background: "#fff",
              color: "#0d6efd",
              cursor: "pointer",
            }}
          >
            Use video frame ({videoFrame})
          </button>
        )}
        {maxFrame !== null && maxFrame !== undefined && (
          <span style={{ fontSize: 12, color: "#6c757d" }}>(golden up to frame {maxFrame})</span>
        )}
      </div>

      {rebuildingPreview && (
        <div style={{ padding: 10, marginBottom: 12, backgroundColor: "#cff4fc", borderRadius: 8, color: "#055160", fontSize: 13 }}>
          Rebuilding golden preview video…
        </div>
      )}

      {error && (
        <div style={{ padding: 10, marginBottom: 12, backgroundColor: "#f8d7da", borderRadius: 8, color: "#721c24", fontSize: 13 }}>
          {error}
        </div>
      )}
      {success && !rebuildingPreview && (
        <div style={{ padding: 10, marginBottom: 12, backgroundColor: "#d4edda", borderRadius: 8, color: "#155724", fontSize: 13 }}>
          {success}
        </div>
      )}

      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
        <thead>
          <tr style={{ backgroundColor: "#e9ecef" }}>
            <th style={{ padding: 10, textAlign: "left", border: "1px solid #dee2e6" }}>Cow ID</th>
            <th style={{ padding: 10, textAlign: "left", border: "1px solid #dee2e6" }}>Current at frame</th>
            <th style={{ padding: 10, textAlign: "left", border: "1px solid #dee2e6" }}>Set new (from frame)</th>
          </tr>
        </thead>
        <tbody>
          {cowIds.map((cowId) => (
            <tr key={cowId}>
              <td style={{ padding: 10, border: "1px solid #dee2e6", fontWeight: 600 }}>{cowId}</td>
              <td style={{ padding: 10, border: "1px solid #dee2e6" }}>
                {labelNameFi(labelsAtFrame[cowId] || "—")}
              </td>
              <td style={{ padding: 10, border: "1px solid #dee2e6" }}>
                <select
                  value={labelsAtFrame[cowId] || ""}
                  disabled={busyCowId === cowId || rebuildingPreview}
                  onChange={(e) => handleLabelChange(cowId, e.target.value)}
                  style={{
                    width: "100%",
                    maxWidth: 280,
                    padding: "6px 8px",
                    borderRadius: 6,
                    border: "1px solid #ccc",
                    fontSize: 13,
                  }}
                >
                  {BEHAVIOR_LABELS.map((l) => (
                    <option key={l.id} value={l.id} title={l.descriptionFi}>
                      {l.nameFi}
                    </option>
                  ))}
                </select>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
