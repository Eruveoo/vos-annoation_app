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
      <h4 style={{ marginTop: 0, marginBottom: 12 }}>
        Assign IDs to detected masks
      </h4>
      <p style={{ fontSize: 12, color: "#666", marginBottom: 12 }}>
        Review and change IDs if needed (e.g., to continue from a previous video
        segment). Uncheck to delete a mask.
      </p>

      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: 14,
          }}
        >
          <thead>
            <tr style={{ backgroundColor: "#f5f5f5" }}>
              <th style={{ padding: 8, textAlign: "left", border: "1px solid #ddd" }}>
                Mask #
              </th>
              <th style={{ padding: 8, textAlign: "left", border: "1px solid #ddd" }}>
                Auto-assigned ID
              </th>
              <th style={{ padding: 8, textAlign: "left", border: "1px solid #ddd" }}>
                Final ID
              </th>
              <th style={{ padding: 8, textAlign: "center", border: "1px solid #ddd" }}>
                Delete?
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
                    backgroundColor: isDeleted ? "#ffe6e6" : "white",
                  }}
                >
                  <td style={{ padding: 8, border: "1px solid #ddd" }}>
                    {maskIndex}
                  </td>
                  <td style={{ padding: 8, border: "1px solid #ddd" }}>
                    {autoId}
                  </td>
                  <td style={{ padding: 8, border: "1px solid #ddd" }}>
                    <input
                      type="number"
                      min="1"
                      value={isDeleted ? "" : finalId}
                      onChange={(e) => handleIdChange(maskIndex, e.target.value)}
                      disabled={isDeleted}
                      style={{
                        width: 80,
                        padding: 4,
                        border: "1px solid #ccc",
                        borderRadius: 4,
                      }}
                    />
                  </td>
                  <td style={{ padding: 8, textAlign: "center", border: "1px solid #ddd" }}>
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
