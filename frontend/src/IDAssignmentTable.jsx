import React from "react";

/**
 * Component for editing ID assignments
 * @param {Array} maskAssignments - Array of {mask_index, auto_assigned_id}
 * @param {Object} idMapping - Current mapping of mask_index -> final_id
 * @param {Function} onMappingChange - Callback when mapping changes
 */
export default function IDAssignmentTable({
  maskAssignments,
  idMapping,
  onMappingChange,
}) {
  // Store the last known Final ID for each mask before deletion
  // This allows us to restore the user's custom ID when un-deleting
  const [lastKnownIds, setLastKnownIds] = React.useState({});

  const handleIdChange = (maskIndex, newId) => {
    const updated = { ...idMapping };
    const id = parseInt(newId, 10);
    
    if (isNaN(id) || id < 1) {
      // Invalid ID - remove from mapping (will be treated as deleted)
      // But preserve the last known ID for restoration
      if (updated[maskIndex] !== undefined) {
        setLastKnownIds(prev => ({ ...prev, [maskIndex]: updated[maskIndex] }));
      }
      delete updated[maskIndex];
    } else {
      updated[maskIndex] = id;
      // Update last known ID when user changes it
      setLastKnownIds(prev => ({ ...prev, [maskIndex]: id }));
    }
    
    onMappingChange(updated);
  };

  const handleDeleteToggle = (maskIndex) => {
    const updated = { ...idMapping };
    if (updated[maskIndex] !== undefined) {
      // Deleting: preserve the current ID for restoration
      setLastKnownIds(prev => ({ ...prev, [maskIndex]: updated[maskIndex] }));
      delete updated[maskIndex];
    } else {
      // Restoring: use the last known Final ID if available, otherwise use auto-assigned ID
      const assignment = maskAssignments.find(
        (m) => m.mask_index === maskIndex
      );
      if (assignment) {
        // Prefer the user's last known Final ID, fallback to auto-assigned ID
        const restoreId = lastKnownIds[maskIndex] ?? assignment.auto_assigned_id;
        updated[maskIndex] = restoreId;
      }
    }
    onMappingChange(updated);
  };

  return (
    <div>
      <style>{`
        /* Ensure number inputs are visually centered in WebKit browsers (Chrome/Safari) */
        .vos-id-input::-webkit-outer-spin-button,
        .vos-id-input::-webkit-inner-spin-button {
          -webkit-appearance: none;
          margin: 0;
        }
      `}</style>
      <h4 style={{ marginTop: 0, marginBottom: 12 }}>
        Assign IDs to detected masks
      </h4>

      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: 14,
            tableLayout: "fixed",
          }}
        >
          <colgroup>
            <col style={{ width: 110 }} />
            <col style={{ width: 140 }} />
            <col style={{ width: 80 }} />
          </colgroup>
          <thead>
            <tr style={{ backgroundColor: "#f5f5f5" }}>
              <th style={{ padding: "8px 6px", textAlign: "center", border: "1px solid #ddd" }}>
                Suggested
              </th>
              <th style={{ padding: "8px 6px", textAlign: "center", border: "1px solid #ddd" }}>
                ID
              </th>
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
