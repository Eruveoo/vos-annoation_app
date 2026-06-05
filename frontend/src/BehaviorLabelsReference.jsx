import React from "react";
import {
  BEHAVIOR_LABELS_ACTIVITY,
  BEHAVIOR_LABELS_LABEL2,
  BEHAVIOR_LABELS_LABEL3,
  BEHAVIOR_DIMENSION_TITLES,
} from "./behaviorLabels.js";

function LabelSection({ title, labels, showGroups = false }) {
  const items = labels.filter((l) => l.id !== "not_visible" && l.id !== "not_seen");

  return (
    <div style={{ marginBottom: 28 }}>
      <h4 style={{ margin: "0 0 12px", fontSize: 16, color: "#212529" }}>{title}</h4>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {items.map((label, index) => (
          <div
            key={label.id}
            style={{
              padding: 14,
              borderRadius: 10,
              border: "1px solid #f1f3f5",
              backgroundColor: "#f8f9fa",
            }}
          >
            {showGroups && label.groupFi && (
              <div style={{ fontSize: 11, fontWeight: 600, color: "#6c757d", marginBottom: 4 }}>
                {label.groupFi}
              </div>
            )}
            <div style={{ fontSize: 14, fontWeight: 700, color: "#212529", marginBottom: 6 }}>
              {index + 1}. {label.nameFi}
            </div>
            <div style={{ fontSize: 13, color: "#495057", lineHeight: 1.55 }}>{label.descriptionFi}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Reference list of all three behaviour label sets. */
export default function BehaviorLabelsReference() {
  return (
    <div
      style={{
        marginTop: 24,
        padding: 20,
        backgroundColor: "#fff",
        borderRadius: 12,
        border: "1px solid #e9ecef",
      }}
    >
      <h3 style={{ marginTop: 0, marginBottom: 8 }}>Label reference</h3>
      <p style={{ fontSize: 13, color: "#6c757d", marginTop: 0, marginBottom: 20, lineHeight: 1.5 }}>
        Each cow has three parallel labels. Label 1 is required; Labels 2 and 3 can stay &quot;Ei valittu&quot;.
      </p>

      <LabelSection title={BEHAVIOR_DIMENSION_TITLES.activity} labels={BEHAVIOR_LABELS_ACTIVITY} />
      <LabelSection
        title={BEHAVIOR_DIMENSION_TITLES.label2}
        labels={BEHAVIOR_LABELS_LABEL2}
        showGroups
      />
      <LabelSection
        title={BEHAVIOR_DIMENSION_TITLES.label3}
        labels={BEHAVIOR_LABELS_LABEL3}
        showGroups
      />
    </div>
  );
}
