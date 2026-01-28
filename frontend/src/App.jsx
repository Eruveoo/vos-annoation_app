import React, { useState, useEffect } from "react";
import Frame0Preview from "./Frame0Preview.jsx";
import IDAssignmentTable from "./IDAssignmentTable.jsx";
import VideoPlayer from "./VideoPlayer.jsx";
import ProgressDisplay from "./ProgressDisplay.jsx";
import CorrectionWorkflow from "./CorrectionWorkflow.jsx";
import {
  initVideo,
  applyInitIds,
  testConnection,
  trackFrames,
  commitFrames,
  getProgress,
  getTrackedVideoUrl,
  getGoldenVideoUrl,
  downloadGolden,
  matchInitIds,
  previewInitUpdate,
} from "./api.js";

export default function App() {
  const [videoPath, setVideoPath] = useState("video_sample_5_min.mp4");
  const [prompt, setPrompt] = useState("cow");
  
  const [runId, setRunId] = useState(null);
  const [frame0Image, setFrame0Image] = useState(null);
  const [maskAssignments, setMaskAssignments] = useState([]);
  const [idMapping, setIdMapping] = useState({});
  
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [connectionStatus, setConnectionStatus] = useState(null);
  const [previousMaskFile, setPreviousMaskFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // Tracking state
  const [nFrames, setNFrames] = useState(50);
  const [autoResetInterval, setAutoResetInterval] = useState("");
  const [trackedVideoUrl, setTrackedVideoUrl] = useState(null);
  const [goldenVideoUrl, setGoldenVideoUrl] = useState(null);
  const [progress, setProgress] = useState({
    processed: null,
    total: null,
    percent: 0,
    fps: null,
    lastChunkSeedIdx: null,
    goldenMaxIdx: null, // For first tracking, use this as seed
  });

  // Test backend connection on mount
  useEffect(() => {
    testConnection().then((connected) => {
      setConnectionStatus(connected);
      if (!connected) {
        setError(
          "⚠️ Cannot connect to backend. Make sure: 1) Backend is running, " +
          "2) SSH port forwarding is active (ssh -N -L 12212:r02g03.bullx:12212 gregormi@puhti.csc.fi)"
        );
      }
    });
  }, []);

  // Refresh progress periodically when runId is set
  useEffect(() => {
    if (!runId) return;

    let lastProcessed = null;
    let pollInterval = 30000; // Start with 30 seconds (less frequent)

    const refreshProgress = async () => {
      try {
        const prog = await getProgress(runId);
        const currentProcessed = prog.golden_processed;
        
        setProgress({
          processed: currentProcessed,
          total: prog.total_frames,
          percent: prog.golden_percent || 0,
          fps: prog.fps || null,
          lastChunkSeedIdx: prog.last_chunk_seed_idx || null,
          goldenMaxIdx: prog.golden_max_idx !== null && prog.golden_max_idx !== undefined 
            ? prog.golden_max_idx 
            : null,
        });

        // Only update golden video URL if the number of processed frames actually changed
        // This prevents unnecessary video reloads
        if (currentProcessed > 0 && currentProcessed !== lastProcessed) {
          setGoldenVideoUrl(getGoldenVideoUrl(runId) + `?t=${Date.now()}`);
          lastProcessed = currentProcessed;
        }
      } catch (e) {
        console.error("Failed to refresh progress:", e);
      }
    };

    // Initial refresh
    refreshProgress();
    
    // Refresh progress less frequently - every 30 seconds when idle
    // This significantly reduces server load while still keeping progress reasonably up-to-date
    // Progress is also refreshed manually after track/commit operations
    const interval = setInterval(refreshProgress, pollInterval);
    return () => clearInterval(interval);
  }, [runId]);

  // Initialize video and get SAM masks
  async function handleInit() {
    if (!videoPath.trim()) {
      setError("Please enter a video path");
      return;
    }

    setBusy(true);
    setError("");
    setSuccess("");
    setRunId(null);
    setFrame0Image(null);
    setMaskAssignments([]);
    setIdMapping({});

    try {
      console.log("Starting initialization...", { videoPath: videoPath.trim(), prompt: prompt.trim() });
      const result = await initVideo(videoPath.trim(), prompt.trim());
      console.log("Init successful!", result);
      
      setRunId(result.run_id);
      setFrame0Image(result.image);
      setMaskAssignments(result.mask_assignments);
      
      // Initialize mapping with auto-assigned IDs
      const initialMapping = {};
      result.mask_assignments.forEach((assignment) => {
        initialMapping[assignment.mask_index] = assignment.auto_assigned_id;
      });
      setIdMapping(initialMapping);
      
      setSuccess(`Initialized! Found ${result.mask_assignments.length} masks. Review and assign IDs below.`);
    } catch (e) {
      console.error("Init failed:", e);
      setError(`Failed to initialize: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  // Update preview when ID mapping changes (debounced)
  useEffect(() => {
    if (!runId || !frame0Image || Object.keys(idMapping).length === 0 || busy) {
      return;
    }

    // Debounce preview updates
    const timeoutId = setTimeout(async () => {
      try {
        const result = await previewInitUpdate(runId, idMapping);
        setFrame0Image(result.image);
      } catch (e) {
        console.error("Failed to update preview:", e);
      }
    }, 300); // 300ms debounce

    return () => clearTimeout(timeoutId);
  }, [runId, idMapping, busy]);

  // Apply ID mappings
  async function handleApplyIds() {
    if (!runId) {
      setError("No run_id. Please initialize first.");
      return;
    }

    // Filter out deleted masks (undefined values)
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
      const result = await applyInitIds(runId, validMapping);
      setSuccess(
        `✅ IDs applied successfully! Run ID: ${result.run_id}, Objects: ${result.n_ids}`
      );
      // Refresh progress after initialization
      const prog = await getProgress(runId);
      setProgress({
        processed: prog.golden_processed,
        total: prog.total_frames,
        percent: prog.golden_percent || 0,
        fps: prog.fps || null,
        lastChunkSeedIdx: prog.last_chunk_seed_idx || null,
        goldenMaxIdx: prog.golden_max_idx !== null && prog.golden_max_idx !== undefined 
          ? prog.golden_max_idx 
          : null,
      });
      
      // Set golden video URL immediately after initialization
      // The backend creates golden_preview.mp4 during initialization (frame 0)
      if (prog.golden_processed > 0) {
        setGoldenVideoUrl(getGoldenVideoUrl(runId) + `?t=${Date.now()}`);
      }
    } catch (e) {
      setError(`Failed to apply IDs: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  // Track frames
  async function handleTrack() {
    if (!runId) {
      setError("No run_id. Please initialize first.");
      return;
    }

    setBusy(true);
    setError("");
    setSuccess("");

    try {
      const resetVal =
        autoResetInterval.trim() !== ""
          ? parseInt(autoResetInterval, 10)
          : null;
      if (resetVal !== null && (isNaN(resetVal) || resetVal < 1)) {
        throw new Error("Auto-reset interval must be a positive number");
      }

      setSuccess("Tracking started... This may take a while.");
      const result = await trackFrames(runId, nFrames, resetVal);

      // Refresh progress
      const prog = await getProgress(runId);
      setProgress({
        processed: prog.golden_processed,
        total: prog.total_frames,
        percent: prog.golden_percent || 0,
        fps: prog.fps || null,
        lastChunkSeedIdx: prog.last_chunk_seed_idx || null,
        goldenMaxIdx: prog.golden_max_idx !== null && prog.golden_max_idx !== undefined 
          ? prog.golden_max_idx 
          : null,
      });

      // Update tracked video URL with a small delay to ensure file is ready
      // Add timestamp to prevent caching
      setTimeout(() => {
        setTrackedVideoUrl(getTrackedVideoUrl(runId) + `?t=${Date.now()}`);
      }, 500);

      setSuccess(
        `✅ Tracking complete! Tracked ${nFrames} frames. Review the tracked video below.`
      );
    } catch (e) {
      setError(`Failed to track: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  // Commit frames
  async function handleCommit() {
    if (!runId) {
      setError("No run_id. Please initialize first.");
      return;
    }

    setBusy(true);
    setError("");
    setSuccess("");

    try {
      await commitFrames(runId);

      // Refresh progress and golden video
      const prog = await getProgress(runId);
      setProgress({
        processed: prog.golden_processed,
        total: prog.total_frames,
        percent: prog.golden_percent || 0,
        fps: prog.fps || null,
        lastChunkSeedIdx: prog.last_chunk_seed_idx || null,
        goldenMaxIdx: prog.golden_max_idx !== null && prog.golden_max_idx !== undefined 
          ? prog.golden_max_idx 
          : null,
      });
      setGoldenVideoUrl(getGoldenVideoUrl(runId) + `?t=${Date.now()}`);

      setSuccess(
        `✅ Committed! Progress: ${prog.golden_processed}/${prog.total_frames} frames (${prog.golden_percent.toFixed(1)}%)`
      );
    } catch (e) {
      setError(`Failed to commit: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      style={{
        fontFamily: "system-ui, sans-serif",
        padding: 24,
        maxWidth: 1400,
        margin: "0 auto",
      }}
    >
      <h1 style={{ marginTop: 0 }}>VOS Annotation App</h1>

      {/* Connection status */}
      {connectionStatus !== null && (
        <div
          style={{
            padding: 8,
            marginBottom: 16,
            backgroundColor: connectionStatus ? "#efe" : "#fee",
            border: `1px solid ${connectionStatus ? "#cfc" : "#fcc"}`,
            borderRadius: 4,
            fontSize: 14,
          }}
        >
          {connectionStatus ? "✅ Backend connected" : "❌ Backend not reachable"}
        </div>
      )}

      {/* Video selection */}
      <div style={{ marginBottom: 24, padding: 16, backgroundColor: "#f9f9f9", borderRadius: 8 }}>
        <h3 style={{ marginTop: 0 }}>1. Load Video</h3>
        <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 300 }}>
            <label style={{ display: "block", marginBottom: 4, fontSize: 14, fontWeight: 500 }}>
              Video path (on server):
            </label>
            <input
              type="text"
              value={videoPath}
              onChange={(e) => setVideoPath(e.target.value)}
              placeholder="video_sample_5_min.mp4"
              disabled={busy}
              style={{
                width: "100%",
                padding: 8,
                border: "1px solid #ccc",
                borderRadius: 4,
                fontSize: 14,
              }}
            />
          </div>
          <div style={{ minWidth: 200 }}>
            <label style={{ display: "block", marginBottom: 4, fontSize: 14, fontWeight: 500 }}>
              Text prompt:
            </label>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="cow"
              disabled={busy}
              style={{
                width: "100%",
                padding: 8,
                border: "1px solid #ccc",
                borderRadius: 4,
                fontSize: 14,
              }}
            />
          </div>
          <button
            onClick={handleInit}
            disabled={busy || !videoPath.trim()}
            style={{
              padding: "8px 16px",
              fontSize: 14,
              fontWeight: 500,
              backgroundColor: busy ? "#ccc" : "#007bff",
              color: "white",
              border: "none",
              borderRadius: 4,
              cursor: busy ? "not-allowed" : "pointer",
            }}
          >
            {busy ? "Loading..." : "🆕 Load video + run SAM"}
          </button>
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

      {/* Frame 0 preview and ID assignment */}
      {frame0Image && maskAssignments.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <h3>2. Review and Assign IDs</h3>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 400px",
              gap: 24,
              alignItems: "flex-start",
            }}
          >
            {/* Left: Frame 0 preview */}
            <div>
              <Frame0Preview imageDataUrl={frame0Image} />
            </div>

            {/* Right: ID assignment table */}
            <div>
              {/* Option to match IDs from previous masks */}
              <div style={{ marginBottom: 16, padding: 12, backgroundColor: "#f0f0f0", borderRadius: 4 }}>
                <label style={{ display: "block", marginBottom: 8, fontSize: 14, fontWeight: 500 }}>
                  Match IDs from previous video (optional):
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
                    
                    if (busy || !runId) {
                      return;
                    }
                    
                    const files = Array.from(e.dataTransfer.files);
                    const pngFile = files.find(f => f.name.toLowerCase().endsWith('.png'));
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
                    borderRadius: 4,
                    backgroundColor: isDragging ? "#e8f4f8" : "#fff",
                    textAlign: "center",
                    cursor: (busy || !runId) ? "not-allowed" : "pointer",
                    transition: "all 0.2s",
                    marginBottom: 8,
                  }}
                >
                  {previousMaskFile ? (
                    <div style={{ fontSize: 12, color: "#060" }}>
                      ✓ Selected: {previousMaskFile.name}
                    </div>
                  ) : (
                    <div style={{ fontSize: 12, color: "#666" }}>
                      {isDragging ? "Drop PNG file here" : "Drag & drop PNG file here or click to browse"}
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
                      cursor: (busy || !runId) ? "not-allowed" : "pointer",
                      zIndex: 1,
                    }}
                  />
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
                  <button
                    onClick={async () => {
                      if (!runId) {
                        setError("No run_id. Please initialize first.");
                        return;
                      }
                      if (!previousMaskFile) {
                        setError("Please select a mask file first.");
                        return;
                      }
                      
                      setBusy(true);
                      setError("");
                      setSuccess("");
                      
                      try {
                        const result = await matchInitIds(runId, previousMaskFile);
                        
                        // Update mask assignments with matched IDs
                        setMaskAssignments(result.mask_assignments);
                        
                        // Update ID mapping to use matched IDs
                        const newMapping = {};
                        result.mask_assignments.forEach((assignment) => {
                          newMapping[assignment.mask_index] = assignment.matched_id;
                        });
                        setIdMapping(newMapping);
                        
                        // Update preview image
                        if (result.image) {
                          setFrame0Image(result.image);
                        }
                        
                        setSuccess(
                          `✅ Matched ${result.matched_count}/${result.total_count} masks to previous IDs!`
                        );
                      } catch (e) {
                        setError(`Failed to match IDs: ${e.message}`);
                      } finally {
                        setBusy(false);
                      }
                    }}
                    disabled={busy || !runId || !previousMaskFile}
                    style={{
                      padding: "6px 12px",
                      fontSize: 12,
                      fontWeight: 500,
                      backgroundColor: (busy || !runId || !previousMaskFile) ? "#ccc" : "#17a2b8",
                      color: "white",
                      border: "none",
                      borderRadius: 4,
                      cursor: (busy || !runId || !previousMaskFile) ? "not-allowed" : "pointer",
                    }}
                  >
                    🔗 Match IDs
                  </button>
                  {previousMaskFile && (
                    <button
                      onClick={() => {
                        setPreviousMaskFile(null);
                        // Reset file input
                        const fileInput = document.querySelector('input[type="file"][accept=".png"]');
                        if (fileInput) {
                          fileInput.value = '';
                        }
                      }}
                      disabled={busy}
                      style={{
                        padding: "6px 12px",
                        fontSize: 12,
                        backgroundColor: "#dc3545",
                        color: "white",
                        border: "none",
                        borderRadius: 4,
                        cursor: busy ? "not-allowed" : "pointer",
                      }}
                    >
                      ✕ Clear
                    </button>
                  )}
                </div>
                <div style={{ fontSize: 11, color: "#666", marginTop: 4 }}>
                  Upload the last frame mask PNG from a previous session (e.g., 00123.png from golden/Annotations/video1/)
                </div>
              </div>
              
              <IDAssignmentTable
                maskAssignments={maskAssignments}
                idMapping={idMapping}
                onMappingChange={setIdMapping}
              />
              <div style={{ marginTop: 16 }}>
                <button
                  onClick={handleApplyIds}
                  disabled={busy}
                  style={{
                    padding: "10px 20px",
                    fontSize: 14,
                    fontWeight: 500,
                    backgroundColor: busy ? "#ccc" : "#28a745",
                    color: "white",
                    border: "none",
                    borderRadius: 4,
                    cursor: busy ? "not-allowed" : "pointer",
                    width: "100%",
                  }}
                >
                  {busy ? "Applying..." : "✅ Apply IDs and complete initialization"}
                </button>
              </div>
              {runId && (
                <div style={{ marginTop: 12, fontSize: 12, color: "#666" }}>
                  Run ID: <code>{runId}</code>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tracking and Golden Preview Section */}
      {runId && (
        <div style={{ marginTop: 32 }}>
          <h3>3. Track and Commit</h3>

          {/* Progress Display */}
          <div style={{ marginBottom: 24, padding: 16, backgroundColor: "#f9f9f9", borderRadius: 8 }}>
            <ProgressDisplay
              processed={progress.processed}
              total={progress.total}
              percent={progress.percent}
            />
          </div>

          {/* Tracking Controls */}
          <div style={{ marginBottom: 24, padding: 16, backgroundColor: "#f9f9f9", borderRadius: 8 }}>
            <h4 style={{ marginTop: 0 }}>Track New Frames</h4>
            <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
              <div>
                <label style={{ display: "block", marginBottom: 4, fontSize: 14, fontWeight: 500 }}>
                  N new frames to track:
                </label>
                <input
                  type="number"
                  min="1"
                  max="5000"
                  value={nFrames}
                  onChange={(e) => setNFrames(parseInt(e.target.value, 10) || 50)}
                  disabled={busy}
                  style={{
                    width: 120,
                    padding: 8,
                    border: "1px solid #ccc",
                    borderRadius: 4,
                    fontSize: 14,
                  }}
                />
              </div>
              <div>
                <label style={{ display: "block", marginBottom: 4, fontSize: 14, fontWeight: 500 }}>
                  Auto-reset interval (optional):
                </label>
                <input
                  type="number"
                  min="1"
                  value={autoResetInterval}
                  onChange={(e) => setAutoResetInterval(e.target.value)}
                  disabled={busy}
                  placeholder="e.g., 10"
                  style={{
                    width: 120,
                    padding: 8,
                    border: "1px solid #ccc",
                    borderRadius: 4,
                    fontSize: 14,
                  }}
                />
                <div style={{ fontSize: 11, color: "#666", marginTop: 4 }}>
                  Reinitialize with SAM every K frames
                </div>
              </div>
              <button
                onClick={handleTrack}
                disabled={busy || !runId}
                style={{
                  padding: "8px 16px",
                  fontSize: 14,
                  fontWeight: 500,
                  backgroundColor: busy ? "#ccc" : "#007bff",
                  color: "white",
                  border: "none",
                  borderRadius: 4,
                  cursor: busy ? "not-allowed" : "pointer",
                }}
              >
                {busy ? "Tracking..." : "Track + render chunk"}
              </button>
              <button
                onClick={handleCommit}
                disabled={busy || !runId}
                style={{
                  padding: "8px 16px",
                  fontSize: 14,
                  fontWeight: 500,
                  backgroundColor: busy ? "#ccc" : "#28a745",
                  color: "white",
                  border: "none",
                  borderRadius: 4,
                  cursor: busy ? "not-allowed" : "pointer",
                }}
              >
                {busy ? "Committing..." : "✅ Commit chunk to golden"}
              </button>
            </div>
          </div>

          {/* Video Previews */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 24,
              marginBottom: 24,
            }}
          >
            <div>
              <VideoPlayer
                videoUrl={trackedVideoUrl}
                label="Tracked chunk preview (current)"
                height={480}
              />
            </div>
            <div>
              <VideoPlayer
                videoUrl={goldenVideoUrl}
                label="Golden preview (committed so far)"
                height={480}
              />
              <button
                onClick={async () => {
                  if (!runId) {
                    setError("No run_id. Please initialize first.");
                    return;
                  }
                  setBusy(true);
                  setError("");
                  setSuccess("");
                  try {
                    await downloadGolden(runId);
                    setSuccess("✅ Golden folder downloaded successfully!");
                  } catch (e) {
                    setError(`Failed to download golden folder: ${e.message}`);
                  } finally {
                    setBusy(false);
                  }
                }}
                disabled={busy || !runId || !goldenVideoUrl}
                style={{
                  padding: "8px 16px",
                  fontSize: 14,
                  fontWeight: 500,
                  backgroundColor: (busy || !runId || !goldenVideoUrl) ? "#ccc" : "#17a2b8",
                  color: "white",
                  border: "none",
                  borderRadius: 4,
                  cursor: (busy || !runId || !goldenVideoUrl) ? "not-allowed" : "pointer",
                  marginTop: 8,
                  width: "100%",
                }}
              >
                {busy ? "Downloading..." : "💾 Download golden folder"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Correction Workflow */}
      {runId && (
        <div style={{ marginTop: 32 }}>
          <CorrectionWorkflow
            runId={runId}
            trackedVideoUrl={trackedVideoUrl}
            fps={progress.fps}
            seedIdx={progress.lastChunkSeedIdx !== null 
              ? progress.lastChunkSeedIdx 
              : progress.goldenMaxIdx !== null 
                ? progress.goldenMaxIdx 
                : 0} // Use lastChunkSeedIdx if available, otherwise goldenMaxIdx, fallback to 0
            onCorrectionApplied={async () => {
              // Refresh progress after correction
              const prog = await getProgress(runId);
              setProgress({
                processed: prog.golden_processed,
                total: prog.total_frames,
                percent: prog.golden_percent || 0,
                fps: prog.fps || null,
                lastChunkSeedIdx: prog.last_chunk_seed_idx || null,
                goldenMaxIdx: prog.golden_max_idx !== null && prog.golden_max_idx !== undefined 
                  ? prog.golden_max_idx 
                  : null,
              });
              setGoldenVideoUrl(getGoldenVideoUrl(runId) + `?t=${Date.now()}`);
            }}
          />
        </div>
      )}
    </div>
  );
}
