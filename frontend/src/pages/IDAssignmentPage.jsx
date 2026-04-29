import React, { useState, useEffect } from "react";
import Frame0Preview from "../Frame0Preview.jsx";
import IDAssignmentTable from "../IDAssignmentTable.jsx";
import { matchInitIds, previewInitUpdate } from "../api.js";

export default function IDAssignmentPage({ runId, frame0Image, maskAssignments, onIdsApplied }) {
  const [idMapping, setIdMapping] = useState({});
  const [currentFrame0Image, setCurrentFrame0Image] = useState(frame0Image);
  const [previousMaskFile, setPreviousMaskFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Initialize mapping with auto-assigned IDs
  useEffect(() => {
    if (maskAssignments.length > 0 && Object.keys(idMapping).length === 0) {
      const initialMapping = {};
      maskAssignments.forEach((assignment) => {
        initialMapping[assignment.mask_index] = assignment.auto_assigned_id;
      });
      setIdMapping(initialMapping);
    }
  }, [maskAssignments]);

  // Update preview when ID mapping changes (debounced)
  useEffect(() => {
    if (!runId || !currentFrame0Image || Object.keys(idMapping).length === 0 || busy) {
      return;
    }

    const timeoutId = setTimeout(async () => {
      try {
        const result = await previewInitUpdate(runId, idMapping);
        setCurrentFrame0Image(result.image);
      } catch (e) {
        console.error("Failed to update preview:", e);
      }
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [runId, idMapping, busy]);

  const handleMatchIds = async () => {
    if (!previousMaskFile) {
      setError("Please select a mask file first.");
      return;
    }

    setBusy(true);
    setError("");
    setSuccess("");

    try {
      const result = await matchInitIds(runId, previousMaskFile);
      const newMapping = {};
      result.mask_assignments.forEach((assignment) => {
        newMapping[assignment.mask_index] = assignment.matched_id;
      });
      setIdMapping(newMapping);
      if (result.image) {
        setCurrentFrame0Image(result.image);
      }
      setSuccess(`Matched ${result.matched_count}/${result.total_count}`);
    } catch (e) {
      setError(`Failed to match IDs: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%)",
        padding: "32px 24px",
      }}
    >
      <div style={{ maxWidth: 1400, margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
          <div style={{ fontSize: 28, fontWeight: 700, letterSpacing: -0.3, color: "#212529" }}>
            VOS Annotation App
          </div>
          {runId && (
            <div style={{ fontSize: 12, color: "#6c757d" }}>
              Run ID: <code>{runId}</code>
            </div>
          )}
        </div>

        <div
          style={{
            backgroundColor: "#fff",
            border: "1px solid #e9ecef",
            borderRadius: 16,
            padding: 32,
            boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
            display: "flex",
            flexDirection: "column",
            gap: 16,
          }}
        >
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
            <h2 style={{ margin: 0 }}>Assign IDs</h2>
          </div>

        {/* Error/Success messages */}
        {error && (
          <div
            style={{
              padding: 12,
              marginBottom: 16,
              backgroundColor: "#f8d7da",
              border: "1px solid #f5c6cb",
              borderRadius: 12,
              color: "#721c24",
            }}
          >
            {error}
          </div>
        )}

        {/* Main content: Preview and Table */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 450px",
            gap: 24,
            alignItems: "flex-start",
          }}
        >
          {/* Left: Frame 0 preview */}
          <div>
            <Frame0Preview imageDataUrl={currentFrame0Image} />
          </div>

          {/* Right: ID assignment table */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Match IDs from previous masks */}
            <div style={{ padding: 16, backgroundColor: "#f8f9fa", borderRadius: 12, border: "1px solid #e9ecef" }}>
              <label style={{ display: "block", marginBottom: 8, fontSize: 14, fontWeight: 700, color: "#495057" }}>
                Previous IDs (optional)
              </label>
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  if (!busy && runId) {
                    setIsDragging(true);
                  }
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIsDragging(false);
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIsDragging(false);
                  if (busy || !runId) return;
                  const files = Array.from(e.dataTransfer.files);
                  const pngFile = files.find((f) => f.name.toLowerCase().endsWith(".png"));
                  if (pngFile) {
                    setPreviousMaskFile(pngFile);
                  } else if (files.length > 0) {
                    setError("Please drop a PNG file.");
                  }
                }}
                style={{
                  position: "relative",
                  padding: 16,
                  border: `2px dashed ${isDragging ? "#17a2b8" : "#ccc"}`,
                  borderRadius: 12,
                  backgroundColor: isDragging ? "#e8f4f8" : "#fff",
                  textAlign: "center",
                  cursor: busy || !runId ? "not-allowed" : "pointer",
                  transition: "all 0.2s",
                  marginBottom: 8,
                }}
              >
                {previousMaskFile ? (
                  <div style={{ fontSize: 12, color: "#495057" }}>{previousMaskFile.name}</div>
                ) : (
                  <div style={{ fontSize: 12, color: "#666" }}>
                    {isDragging ? "Drop PNG here" : "Drop PNG here or click to browse"}
                  </div>
                )}
                <input
                  type="file"
                  accept=".png"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    setPreviousMaskFile(file || null);
                  }}
                  disabled={busy || !runId}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: "100%",
                    opacity: 0,
                    cursor: busy || !runId ? "not-allowed" : "pointer",
                    zIndex: 1,
                  }}
                />
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8, flexWrap: "wrap" }}>
                <button
                  onClick={handleMatchIds}
                  disabled={busy || !runId || !previousMaskFile}
                  style={{
                    padding: "6px 12px",
                    fontSize: 12,
                    fontWeight: 500,
                    backgroundColor: busy || !runId || !previousMaskFile ? "#ccc" : "#17a2b8",
                    color: "white",
                    border: "none",
                    borderRadius: 8,
                    cursor: busy || !runId || !previousMaskFile ? "not-allowed" : "pointer",
                  }}
                >
                  Match IDs
                </button>
                {previousMaskFile && (
                  <button
                    onClick={() => {
                      setPreviousMaskFile(null);
                      const fileInput = document.querySelector('input[type="file"][accept=".png"]');
                      if (fileInput) fileInput.value = "";
                    }}
                    disabled={busy}
                    style={{
                      padding: "6px 12px",
                      fontSize: 12,
                      backgroundColor: "#dc3545",
                      color: "white",
                      border: "none",
                      borderRadius: 8,
                      cursor: busy ? "not-allowed" : "pointer",
                    }}
                  >
                    Clear
                  </button>
                )}
                {success && (
                  <span style={{ fontSize: 12, color: "#6c757d", marginLeft: 4 }}>
                    Matched {success}
                  </span>
                )}
              </div>
            </div>

            <IDAssignmentTable
              maskAssignments={maskAssignments}
              idMapping={idMapping}
              onMappingChange={setIdMapping}
            />
          </div>
        </div>

        {/* Apply button */}
        <button
          onClick={() => onIdsApplied(idMapping)}
          disabled={busy}
          style={{
            position: "sticky",
            bottom: 16,
            padding: "12px 24px",
            fontSize: 16,
            fontWeight: 600,
            backgroundColor: busy ? "#6c757d" : "#28a745",
            color: "white",
            border: "none",
            borderRadius: 12,
            cursor: busy ? "not-allowed" : "pointer",
            width: "100%",
            boxShadow: "0 10px 24px rgba(40, 167, 69, 0.18)",
          }}
        >
          {busy ? "Applying..." : "Apply IDs"}
        </button>
      </div>
      </div>
    </div>
  );
}
