import React, { useState, useEffect, useRef } from "react";
import InteractiveVideoPlayer from "./InteractiveVideoPlayer.jsx";
import Frame0Preview from "./Frame0Preview.jsx";
import InteractiveCanvas from "./InteractiveCanvas.jsx";
import IDAssignmentTable from "./IDAssignmentTable.jsx";
import {
  prepareCorrection,
  previewCorrectionUpdate,
  applyCorrection,
  refineMask,
  addMask,
} from "./api.js";

/**
 * Correction workflow component
 * @param {string} runId - Run ID
 * @param {string} trackedVideoUrl - URL to tracked video
 * @param {number} fps - Video FPS
 * @param {number} seedIdx - Seed frame index
 * @param {Function} onCorrectionApplied - Callback when correction is applied
 */
export default function CorrectionWorkflow({
  runId,
  trackedVideoUrl,
  fps,
  seedIdx,
  onCorrectionApplied,
}) {
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [manualFrameInput, setManualFrameInput] = useState("");
  const [correctionImage, setCorrectionImage] = useState(null);
  const [maskAssignments, setMaskAssignments] = useState([]);
  const [idMapping, setIdMapping] = useState({});
  const [absoluteFrameIdx, setAbsoluteFrameIdx] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  
  // Point-based refinement state
  const [selectedMaskIndex, setSelectedMaskIndex] = useState(null);
  const [refinementPoints, setRefinementPoints] = useState([]); // Array of {x, y, is_positive}
  const [imageDimensions, setImageDimensions] = useState({ width: null, height: null });
  const [refiningMask, setRefiningMask] = useState(false);
  const [mode, setMode] = useState("refine"); // "refine" or "add_mask"

  // Update selected frame when video position changes (for display only)
  const handleFrameChange = (absoluteFrame) => {
    // Only update display, don't change selected frame unless explicitly selected
    setManualFrameInput(String(absoluteFrame));
  };

  // Explicitly select frame from video
  const handleSelectFrame = (absoluteFrame) => {
    setSelectedFrame(absoluteFrame);
    setManualFrameInput(String(absoluteFrame));
    setSuccess(`Frame ${absoluteFrame} selected. Click "Prepare correction" to continue.`);
  };

  // Quick action: correct current frame (select + prepare in one step)
  const handleCorrectCurrentFrame = async (absoluteFrame) => {
    if (!absoluteFrame || absoluteFrame < 1) {
      setError("Invalid frame number");
      return;
    }

    // Set as selected
    setSelectedFrame(absoluteFrame);
    setManualFrameInput(String(absoluteFrame));

    // Immediately prepare correction
    await handlePrepareCorrectionWithFrame(absoluteFrame);
  };

  // Prepare correction with a specific frame (extracted for reuse)
  const handlePrepareCorrectionWithFrame = async (frameIdx) => {
    if (!runId) {
      setError("No run_id available");
      return;
    }

    if (isNaN(frameIdx) || frameIdx < 0) {
      setError("Please select a valid frame (>= 0)");
      return;
    }

    setBusy(true);
    setError("");
    setSuccess("");
    setCorrectionImage(null);
    setMaskAssignments([]);
    setIdMapping({});

    try {
      const result = await prepareCorrection(runId, frameIdx);
      
      setAbsoluteFrameIdx(frameIdx);
      setCorrectionImage(result.image);
      setMaskAssignments(result.mask_assignments);
      
      // Store image dimensions for point coordinate scaling
      if (result.image_width && result.image_height) {
        setImageDimensions({
          width: result.image_width,
          height: result.image_height,
        });
      }
      
      // Initialize mapping with auto-assigned IDs
      const initialMapping = {};
      result.mask_assignments.forEach((assignment) => {
        initialMapping[assignment.mask_index] = assignment.auto_assigned_id;
      });
      setIdMapping(initialMapping);
      
      // Reset refinement state
      setSelectedMaskIndex(null);
      setRefinementPoints([]);
      
      // If no masks found, default to "add_mask" mode so user can add masks
      if (result.mask_assignments.length === 0) {
        setMode("add_mask");
      }
      
      setSuccess(
        `✅ Frame ${frameIdx} prepared! Found ${result.mask_assignments.length} masks. ${result.mask_assignments.length === 0 ? "You can add masks using point prompts below." : "Review and assign IDs below."}`
      );
    } catch (e) {
      setError(`Failed to prepare correction: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  // Prepare correction
  const handlePrepareCorrection = async () => {
    const frameIdx = selectedFrame !== null ? selectedFrame : parseInt(manualFrameInput, 10);
    await handlePrepareCorrectionWithFrame(frameIdx);
  };

  // Track if we just refined a mask to prevent preview update from overwriting it
  const justRefinedRef = useRef(false);

  // Update preview when ID mapping changes
  useEffect(() => {
    if (!runId || absoluteFrameIdx === null || Object.keys(idMapping).length === 0) {
      return;
    }

    // Don't update preview if we're refining a mask (to avoid conflicts)
    if (refiningMask) {
      return;
    }

    // Don't update preview if we just refined a mask (it already has the correct image)
    if (justRefinedRef.current) {
      justRefinedRef.current = false; // Reset flag after skipping this update
      return;
    }

    // Debounce preview updates
    const timeoutId = setTimeout(async () => {
      try {
        const result = await previewCorrectionUpdate(runId, absoluteFrameIdx, idMapping);
        setCorrectionImage(result.image);
      } catch (e) {
        console.error("Failed to update preview:", e);
      }
    }, 300); // 300ms debounce

    return () => clearTimeout(timeoutId);
  }, [runId, absoluteFrameIdx, idMapping, refiningMask]);

  // Handle adding a point for mask refinement or adding a new mask
  const handleAddPoint = async (x, y, isPositive) => {
    if (!runId || absoluteFrameIdx === null) {
      return;
    }

    // Prevent concurrent requests
    if (refiningMask) {
      console.log("[REFINE] Already processing, ignoring click");
      return;
    }

    setRefiningMask(true);
    setError("");
    setSuccess("");

    try {
      if (mode === "add_mask") {
        // Add a new mask with point prompt
        if (!isPositive) {
          setError("New masks must be created with positive points. Use left click.");
          setRefiningMask(false);
          return;
        }

        const result = await addMask(runId, absoluteFrameIdx, x, y, true);
        
        // Set flag to prevent preview update from overwriting
        justRefinedRef.current = true;
        
        // Update preview image
        setCorrectionImage(result.image);
        
        // Update image dimensions if provided
        if (result.image_width && result.image_height) {
          setImageDimensions({
            width: result.image_width,
            height: result.image_height,
          });
        }
        
        // Update mask assignments with the new mask (from add_mask response)
        if (result.mask_assignments) {
          setMaskAssignments(result.mask_assignments);
          
          // Update ID mapping - preserve existing mappings and add new mask
          const newMapping = { ...idMapping };
          // Find the auto-assigned ID for the new mask
          const newMaskAssignment = result.mask_assignments.find(
            (a) => a.mask_index === result.new_mask_index
          );
          if (newMaskAssignment) {
            // Use max existing ID + 1 for the new mask
            const maxId = Math.max(...Object.values(idMapping).filter(id => id >= 1), 0);
            newMapping[result.new_mask_index] = maxId + 1;
          }
          setIdMapping(newMapping);
        } else {
          // Fallback: manually add the new mask to assignments
          const newAssignment = {
            mask_index: result.new_mask_index,
            auto_assigned_id: result.new_mask_index + 1,
          };
          setMaskAssignments([...maskAssignments, newAssignment]);
          
          // Initialize mapping for new mask (use max existing ID + 1)
          const maxId = Math.max(...Object.values(idMapping).filter(id => id >= 1), 0);
          setIdMapping({ ...idMapping, [result.new_mask_index]: maxId + 1 });
        }
        
        setSuccess(
          `✅ New mask added! Index: ${result.new_mask_index}, Size: ${result.new_mask_size} pixels. Total masks: ${result.total_masks}.`
        );
        
        // Switch back to refine mode after adding
        // Remember the initial point used to create the mask for future refinements
        setMode("refine");
        setSelectedMaskIndex(result.new_mask_index);
        setRefinementPoints([{ x, y, is_positive: true }]); // Keep the initial point
      } else {
        // Refine existing mask
        if (selectedMaskIndex === null) {
          setError("Please select a mask to refine first.");
          setRefiningMask(false);
          return;
        }

        // Add point to local state
        const newPoint = { x, y, is_positive: isPositive };
        const updatedPoints = [...refinementPoints, newPoint];
        setRefinementPoints(updatedPoints);

        const result = await refineMask(runId, absoluteFrameIdx, selectedMaskIndex, updatedPoints);
        
        // Set flag to prevent preview update from overwriting the refined mask
        justRefinedRef.current = true;
        
        // Update preview image with refined mask
        setCorrectionImage(result.image);
        
        // Update image dimensions if provided
        if (result.image_width && result.image_height) {
          setImageDimensions({
            width: result.image_width,
            height: result.image_height,
          });
        }
        
        if (result.warning) {
          setError(`⚠️ ${result.warning}`);
        }
        // Success message removed - refinement is visible in the preview
      }
    } catch (e) {
      setError(`Failed to ${mode === "add_mask" ? "add mask" : "refine mask"}: ${e.message}`);
      justRefinedRef.current = false; // Reset flag on error
    } finally {
      setRefiningMask(false);
    }
  };

  // Delete a specific point by index
  const handleDeletePoint = async (pointIndex) => {
    if (selectedMaskIndex === null || !runId || absoluteFrameIdx === null) {
      return;
    }

    // Prevent concurrent requests
    if (refiningMask) {
      console.log("[REFINE] Already refining, ignoring delete");
      return;
    }

    // Remove the point from local state
    const updatedPoints = refinementPoints.filter((_, idx) => idx !== pointIndex);
    setRefinementPoints(updatedPoints);

    // If no points left, reset to original correction image
    if (updatedPoints.length === 0) {
      setRefiningMask(true);
      setError("");
      setSuccess("");
      try {
        const result = await prepareCorrection(runId, absoluteFrameIdx);
        setCorrectionImage(result.image);
        setSuccess(`Point deleted. All refinement points cleared.`);
      } catch (e) {
        setError(`Failed to reset: ${e.message}`);
      } finally {
        setRefiningMask(false);
      }
      return;
    }

    // Re-refine with remaining points
    setRefiningMask(true);
    setError("");
    setSuccess("");
    try {
      const result = await refineMask(runId, absoluteFrameIdx, selectedMaskIndex, updatedPoints);
      
      // Set flag to prevent preview update from overwriting the refined mask
      justRefinedRef.current = true;
      
      // Update preview image with refined mask
      setCorrectionImage(result.image);
      
      // Update image dimensions if provided
      if (result.image_width && result.image_height) {
        setImageDimensions({
          width: result.image_width,
          height: result.image_height,
        });
      }
      
      if (result.warning) {
        setError(`⚠️ ${result.warning}`);
      } else {
        // Success message removed - refinement is visible in the preview
      }
    } catch (e) {
      setError(`Failed to refine mask after deletion: ${e.message}`);
      // Restore the point on error
      setRefinementPoints(refinementPoints);
      justRefinedRef.current = false; // Reset flag on error
    } finally {
      setRefiningMask(false);
    }
  };

  // Clear refinement points
  const handleClearPoints = async () => {
    if (!runId || absoluteFrameIdx === null) {
      setRefinementPoints([]);
      setSelectedMaskIndex(null);
      return;
    }

    // Reset to original correction image
    setRefiningMask(true);
    setError("");
    setSuccess("");
    setRefinementPoints([]);
    
    try {
      const result = await prepareCorrection(runId, absoluteFrameIdx);
      setCorrectionImage(result.image);
      setSuccess(`All refinement points cleared.`);
    } catch (e) {
      setError(`Failed to clear points: ${e.message}`);
    } finally {
      setRefiningMask(false);
    }
  };

  // Apply correction
  const handleApplyCorrection = async () => {
    if (!runId || absoluteFrameIdx === null) {
      setError("No frame selected for correction");
      return;
    }

    // Filter out deleted masks
    const validMapping = {};
    Object.entries(idMapping).forEach(([maskIndex, finalId]) => {
      if (finalId !== undefined && finalId >= 1) {
        validMapping[maskIndex] = finalId;
      }
    });

    if (Object.keys(validMapping).length === 0) {
      setError("No valid IDs assigned. Please assign at least one ID.");
      return;
    }

    setBusy(true);
    setError("");
    setSuccess("");

    try {
      const result = await applyCorrection(runId, absoluteFrameIdx, validMapping);
      setSuccess(
        `✅ Frame ${absoluteFrameIdx} corrected! Max ID: ${result.max_id}`
      );
      
      // Reset state
      setCorrectionImage(null);
      setMaskAssignments([]);
      setIdMapping({});
      setAbsoluteFrameIdx(null);
      
      // Notify parent to refresh progress
      if (onCorrectionApplied) {
        onCorrectionApplied();
      }
    } catch (e) {
      setError(`Failed to apply correction: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h3>4. Frame Correction</h3>
      
      <div style={{ marginBottom: 24, padding: 16, backgroundColor: "#f9f9f9", borderRadius: 8 }}>
        <h4 style={{ marginTop: 0 }}>Select Frame to Correct</h4>
        <p style={{ fontSize: 14, color: "#666", marginBottom: 16 }}>
          <strong>Method 1:</strong> Scrub through the tracked video below and the current frame will be shown.
          <br />
          <strong>Method 2:</strong> Enter frame number manually.
        </p>

        {/* Interactive video player */}
        {trackedVideoUrl && fps !== null ? (
          <div style={{ marginBottom: 16 }}>
            {seedIdx !== null ? (
              <InteractiveVideoPlayer
                videoUrl={trackedVideoUrl}
                label="Tracked video (scrub through video to find the frame to correct)"
                height={360}
                fps={fps}
                seedIdx={seedIdx}
                onFrameChange={handleFrameChange}
                onSelectFrame={handleSelectFrame}
                onCorrectFrame={handleCorrectCurrentFrame}
                selectedFrame={selectedFrame}
                busy={busy}
              />
            ) : (
              <div style={{ padding: 16, backgroundColor: "#fff3cd", border: "1px solid #ffc107", borderRadius: 4 }}>
                <p style={{ margin: 0, color: "#856404" }}>
                  ⚠️ Seed index not available yet. Please track some frames first, or enter frame number manually below.
                </p>
              </div>
            )}
          </div>
        ) : (
          <div style={{ padding: 16, backgroundColor: "#f5f5f5", border: "1px solid #ccc", borderRadius: 4 }}>
            <p style={{ margin: 0, color: "#666" }}>
              Track some frames first to enable frame correction from video.
            </p>
          </div>
        )}

        {/* Manual frame input */}
        <div style={{ marginBottom: 16 }}>
          {selectedFrame !== null && (
            <div
              style={{
                padding: 12,
                marginBottom: 12,
                backgroundColor: "#e3f2fd",
                border: "1px solid #2196f3",
                borderRadius: 4,
                fontSize: 14,
              }}
            >
              <strong>Selected frame: {selectedFrame}</strong>
            </div>
          )}
          <div style={{ display: "flex", gap: 12, alignItems: "flex-end" }}>
            <div>
              <label style={{ display: "block", marginBottom: 4, fontSize: 14, fontWeight: 500 }}>
                Or enter frame number manually:
              </label>
              <input
                type="number"
                min="0"
                value={manualFrameInput}
                onChange={(e) => {
                  setManualFrameInput(e.target.value);
                  const val = parseInt(e.target.value, 10);
                  if (!isNaN(val)) {
                    setSelectedFrame(val);
                  }
                }}
                disabled={busy}
                placeholder="Frame number"
                style={{
                  width: 150,
                  padding: 8,
                  border: "1px solid #ccc",
                  borderRadius: 4,
                  fontSize: 14,
                }}
              />
            </div>
            <button
              onClick={handlePrepareCorrection}
              disabled={busy || selectedFrame === null}
              style={{
                padding: "8px 16px",
                fontSize: 14,
                fontWeight: 500,
                backgroundColor: busy ? "#ccc" : selectedFrame !== null ? "#ff9800" : "#999",
                color: "white",
                border: "none",
                borderRadius: 4,
                cursor: busy || selectedFrame === null ? "not-allowed" : "pointer",
              }}
            >
              {busy ? "Preparing..." : "🔧 Prepare correction (run SAM)"}
            </button>
          </div>
        </div>
      </div>

      {/* Error/Success messages */}
      {error && (
        <div
          style={{
            padding: 12,
            marginBottom: 16,
            backgroundColor: "#fee",
            border: "1px solid #fcc",
            borderRadius: 4,
            color: "#c00",
          }}
        >
          {error}
        </div>
      )}
      {success && (
        <div
          style={{
            padding: 12,
            marginBottom: 16,
            backgroundColor: "#efe",
            border: "1px solid #cfc",
            borderRadius: 4,
            color: "#060",
          }}
        >
          {success}
        </div>
      )}

      {/* Correction preview and ID assignment */}
      {correctionImage && (
        <div style={{ marginBottom: 24 }}>
          <h4>Review and Assign IDs</h4>
          
          {/* Mode selection and mask selection */}
          <div style={{ marginBottom: 16, padding: 12, backgroundColor: "#f0f0f0", borderRadius: 4 }}>
            <div style={{ marginBottom: 12 }}>
              <label style={{ display: "block", marginBottom: 8, fontSize: 14, fontWeight: 500 }}>
                Mode:
              </label>
              <div style={{ display: "flex", gap: 8 }}>
                <button
                  onClick={() => {
                    setMode("refine");
                    setSelectedMaskIndex(null);
                    setRefinementPoints([]);
                  }}
                  style={{
                    padding: "6px 12px",
                    fontSize: 14,
                    backgroundColor: mode === "refine" ? "#4CAF50" : "#e0e0e0",
                    color: mode === "refine" ? "white" : "#333",
                    border: "none",
                    borderRadius: 4,
                    cursor: "pointer",
                    fontWeight: mode === "refine" ? 600 : 400,
                  }}
                >
                  ✏️ Refine existing mask
                </button>
                <button
                  onClick={() => {
                    setMode("add_mask");
                    setSelectedMaskIndex(null);
                    setRefinementPoints([]);
                  }}
                  style={{
                    padding: "6px 12px",
                    fontSize: 14,
                    backgroundColor: mode === "add_mask" ? "#2196F3" : "#e0e0e0",
                    color: mode === "add_mask" ? "white" : "#333",
                    border: "none",
                    borderRadius: 4,
                    cursor: "pointer",
                    fontWeight: mode === "add_mask" ? 600 : 400,
                  }}
                >
                  ➕ Add new mask
                </button>
              </div>
            </div>

            {mode === "refine" && (
              <>
                <label style={{ display: "block", marginBottom: 8, fontSize: 14, fontWeight: 500 }}>
                  Select mask to refine with points:
                </label>
                <select
                  value={selectedMaskIndex !== null ? selectedMaskIndex : ""}
                  onChange={(e) => {
                    const maskIdx = e.target.value === "" ? null : parseInt(e.target.value, 10);
                    setSelectedMaskIndex(maskIdx);
                    setRefinementPoints([]); // Clear points when switching masks
                  }}
                  style={{
                    padding: "6px 12px",
                    fontSize: 14,
                    border: "1px solid #ccc",
                    borderRadius: 4,
                    minWidth: 200,
                  }}
                >
                  <option value="">None (just assign IDs)</option>
                  {maskAssignments.map((assignment) => (
                    <option key={assignment.mask_index} value={assignment.mask_index}>
                      Mask {assignment.mask_index} (ID: {idMapping[assignment.mask_index] || "?"})
                    </option>
                  ))}
                </select>
                {selectedMaskIndex !== null && (
                  <div style={{ marginTop: 8, fontSize: 12, color: "#666" }}>
                    Click on the image to add points: <span style={{ color: "green" }}>Left = add</span>,{" "}
                    <span style={{ color: "red" }}>Right = remove</span>
                  </div>
                )}
              </>
            )}

            {mode === "add_mask" && (
              <div style={{ marginTop: 8, fontSize: 12, color: "#666" }}>
                <span style={{ color: "#2196F3", fontWeight: 500 }}>Click on the image</span> where a new cow should be detected.
                Use <span style={{ color: "green" }}>left click</span> to add a new mask at that location.
              </div>
            )}
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 400px",
              gap: 24,
              alignItems: "flex-start",
            }}
          >
            {/* Left: Correction preview (interactive if mask selected) */}
            <div>
                     {(mode === "add_mask" || (selectedMaskIndex !== null)) && imageDimensions.width && imageDimensions.height ? (
                       <div style={{ position: "relative", opacity: refiningMask ? 0.6 : 1, pointerEvents: refiningMask ? "none" : "auto" }}>
                         <InteractiveCanvas
                           imageDataUrl={correctionImage}
                           width={imageDimensions.width}
                           height={imageDimensions.height}
                           points={mode === "add_mask" ? [] : refinementPoints}
                           onAddPoint={handleAddPoint}
                           onDeletePoint={mode === "add_mask" ? undefined : handleDeletePoint}
                           onClearPoints={mode === "add_mask" ? undefined : handleClearPoints}
                           key={correctionImage} // Force re-render when image changes
                         />
                  {refiningMask && (
                    <div style={{ 
                      position: "absolute", 
                      top: "50%", 
                      left: "50%", 
                      transform: "translate(-50%, -50%)",
                      backgroundColor: "rgba(0,0,0,0.7)",
                      color: "white",
                      padding: "10px 20px",
                      borderRadius: 4,
                      fontSize: 14,
                      zIndex: 10,
                    }}>
                      Refining mask...
                    </div>
                  )}
                </div>
              ) : (
                <Frame0Preview imageDataUrl={correctionImage} key={correctionImage} />
              )}
            </div>

            {/* Right: ID assignment table */}
            <div>
              <IDAssignmentTable
                maskAssignments={maskAssignments}
                idMapping={idMapping}
                onMappingChange={setIdMapping}
              />
              <div style={{ marginTop: 16 }}>
                <button
                  onClick={handleApplyCorrection}
                  disabled={busy || refiningMask}
                  style={{
                    padding: "10px 20px",
                    fontSize: 14,
                    fontWeight: 500,
                    backgroundColor: busy || refiningMask ? "#ccc" : "#28a745",
                    color: "white",
                    border: "none",
                    borderRadius: 4,
                    cursor: busy || refiningMask ? "not-allowed" : "pointer",
                    width: "100%",
                  }}
                >
                  {busy ? "Applying..." : refiningMask ? "Refining..." : "✅ Apply correction"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
