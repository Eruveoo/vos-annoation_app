import React from "react";

/**
 * Progress display component
 * @param {number} processed - Number of processed frames
 * @param {number} total - Total number of frames
 * @param {number} percent - Progress percentage
 */
export default function ProgressDisplay({ processed, total, percent }) {
  if (processed === null || total === null) {
    return (
      <div style={{ fontSize: 14, color: "#666" }}>
        <strong>Golden progress:</strong> —
      </div>
    );
  }

  return (
    <div>
      <div style={{ marginBottom: 8, fontSize: 14 }}>
        <strong>Golden progress:</strong> {processed}/{total} frames ({percent.toFixed(1)}%)
      </div>
      <div
        style={{
          width: "100%",
          height: 24,
          backgroundColor: "#e0e0e0",
          borderRadius: 4,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${percent}%`,
            height: "100%",
            backgroundColor: "#28a745",
            transition: "width 0.3s ease",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "white",
            fontSize: 12,
            fontWeight: 500,
          }}
        >
          {percent > 5 ? `${percent.toFixed(1)}%` : ""}
        </div>
      </div>
    </div>
  );
}
