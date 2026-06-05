import React from "react";
import {
  BEHAVIOR_DIMENSION_TITLES,
  labelsForInitSelect,
} from "./behaviorLabels.js";

/**
 * Component for editing ID assignments
 */
export default function IDAssignmentTable({
  maskAssignments,
  idMapping,
  onMappingChange,
  showBehavior = false,
  behaviorMapping = {},
  onBehaviorMappingChange,
  behaviorLabel2Mapping = {},
  onBehaviorLabel2MappingChange,
  behaviorLabel3Mapping = {},
  onBehaviorLabel3MappingChange,
}) {
  const [lastKnownIds, setLastKnownIds] = React.useState({});

  const handleIdChange = (maskIndex, newId) => {
    const updated = { ...idMapping };
    const id = parseInt(newId, 10);

    if (isNaN(id) || id < 1) {
      if (updated[maskIndex] !== undefined) {
        setLastKnownIds((prev) => ({ ...prev, [maskIndex]: updated[maskIndex] }));
      }
      delete updated[maskIndex];
    } else {
      updated[maskIndex] = id;
      setLastKnownIds((prev) => ({ ...prev, [maskIndex]: id }));
    }

    onMappingChange(updated);
  };

  const handleDeleteToggle = (maskIndex) => {
    const updated = { ...idMapping };
    if (updated[maskIndex] !== undefined) {
      setLastKnownIds((prev) => ({ ...prev, [maskIndex]: updated[maskIndex] }));
      delete updated[maskIndex];
    } else {
      const assignment = maskAssignments.find((m) => m.mask_index === maskIndex);
      if (assignment) {
        const restoreId = lastKnownIds[maskIndex] ?? assignment.auto_assigned_id;
        updated[maskIndex] = restoreId;
      }
    }
    onMappingChange(updated);
  };

  const behaviorSelect = (maskIndex, dimension, mapping, onChange, defaultId) => (
    <select
      value={mapping[maskIndex] || defaultId}
      disabled={idMapping[maskIndex] === undefined}
      onChange={(e) => {
        if (!onChange) return;
        onChange({ ...mapping, [maskIndex]: e.target.value });
      }}
      style={{
        width: "100%",
        minWidth: 140,
        fontSize: 11,
        padding: "4px 2px",
        borderRadius: 4,
        border: "1px solid #ccc",
      }}
    >
      {labelsForInitSelect(dimension).map((l) => (
        <option key={l.id} value={l.id} title={l.descriptionFi}>
          {l.nameFi}
        </option>
      ))}
    </select>
  );

  return (
    <div>
      <style>{`
        .vos-id-input::-webkit-outer-spin-button,
        .vos-id-input::-webkit-inner-spin-button {
          -webkit-appearance: none;
          margin: 0;
        }
      `}</style>
      <h4 style={{ marginTop: 0, marginBottom: 12 }}>Assign IDs to detected masks</h4>

      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: 13,
            tableLayout: "auto",
          }}
        >
          <thead>
            <tr style={{ backgroundColor: "#f5f5f5" }}>
              <th style={{ padding: "8px 6px", textAlign: "center", border: "1px solid #ddd" }}>
                Suggested
              </th>
              <th style={{ padding: "8px 6px", textAlign: "center", border: "1px solid #ddd" }}>
                ID
              </th>
              {showBehavior && (
                <>
                  <th style={{ padding: "8px 6px", textAlign: "center", border: "1px solid #ddd" }}>
                    {BEHAVIOR_DIMENSION_TITLES.activity}
                  </th>
                  <th style={{ padding: "8px 6px", textAlign: "center", border: "1px solid #ddd" }}>
                    {BEHAVIOR_DIMENSION_TITLES.label2}
                  </th>
                  <th style={{ padding: "8px 6px", textAlign: "center", border: "1px solid #ddd" }}>
                    {BEHAVIOR_DIMENSION_TITLES.label3}
                  </th>
                </>
              )}
              <th style={{ padding: "8px 6px", textAlign: "center", border: "1px solid #ddd" }}>
                Delete
              </th>
            </tr>
          </thead>
          <tbody>
            {maskAssignments.map((assignment) => {
              const maskIndex = assignment.mask_index;
              const autoId = assignment.auto_assigned_id;
              const finalId = idMapping[maskIndex] ?? autoId;
              const isDeleted = idMapping[maskIndex] === undefined;

              return (
                <tr
                  key={maskIndex}
                  style={{
                    backgroundColor: isDeleted ? "#f8f9fa" : "white",
                    color: isDeleted ? "#adb5bd" : "#212529",
                    textDecoration: isDeleted ? "line-through" : "none",
                  }}
                >
                  <td style={{ padding: "8px 6px", border: "1px solid #ddd", textAlign: "center" }}>
                    {autoId}
                  </td>
                  <td style={{ padding: "8px 6px", border: "1px solid #ddd", textAlign: "center" }}>
                    <input
                      className="vos-id-input"
                      type="number"
                      min="1"
                      value={isDeleted ? "" : finalId}
                      onChange={(e) => handleIdChange(maskIndex, e.target.value)}
                      disabled={isDeleted}
                      style={{
                        width: 70,
                        padding: "4px 0",
                        border: "1px solid #ccc",
                        borderRadius: 4,
                        backgroundColor: isDeleted ? "#f1f3f5" : "white",
                        color: isDeleted ? "#adb5bd" : "#212529",
                        textAlign: "center",
                        appearance: "textfield",
                        MozAppearance: "textfield",
                        lineHeight: "20px",
                      }}
                    />
                  </td>
                  {showBehavior && (
                    <>
                      <td style={{ padding: "8px 6px", border: "1px solid #ddd", textAlign: "center" }}>
                        {behaviorSelect(
                          maskIndex,
                          "activity",
                          behaviorMapping,
                          onBehaviorMappingChange,
                          "stand"
                        )}
                      </td>
                      <td style={{ padding: "8px 6px", border: "1px solid #ddd", textAlign: "center" }}>
                        {behaviorSelect(
                          maskIndex,
                          "label2",
                          behaviorLabel2Mapping,
                          onBehaviorLabel2MappingChange,
                          "none"
                        )}
                      </td>
                      <td style={{ padding: "8px 6px", border: "1px solid #ddd", textAlign: "center" }}>
                        {behaviorSelect(
                          maskIndex,
                          "label3",
                          behaviorLabel3Mapping,
                          onBehaviorLabel3MappingChange,
                          "none"
                        )}
                      </td>
                    </>
                  )}
                  <td style={{ padding: "8px 6px", textAlign: "center", border: "1px solid #ddd" }}>
                    <input
                      type="checkbox"
                      checked={isDeleted}
                      onChange={() => handleDeleteToggle(maskIndex)}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
