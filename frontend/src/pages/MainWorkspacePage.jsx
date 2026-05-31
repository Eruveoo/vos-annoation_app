import React, { useState } from "react";
import VideoPlayer from "../VideoPlayer.jsx";
import ProgressDisplay from "../ProgressDisplay.jsx";
import CorrectionWorkflow from "../CorrectionWorkflow.jsx";
import Frame0Preview from "../Frame0Preview.jsx";
import {
  trackFrames,
  commitFrames,
  getProgress,
  getTrackedVideoUrl,
  getGoldenVideoUrl,
  downloadGolden,
  rebuildGoldenPreview,
} from "../api.js";
import BehaviorEditor from "../BehaviorEditor.jsx";
import { ANNOTATION_MODES } from "../behaviorLabels.js";

export default function MainWorkspacePage({ runId, frame0Image, annotationMode, onProgressUpdate }) {
  const isBehaviorMode = annotationMode === ANNOTATION_MODES.BEHAVIOR;
  const [activeTab, setActiveTab] = useState("tracking"); // Default to tracking
  const [nFrames, setNFrames] = useState(50);
  const [autoResetInterval, setAutoResetInterval] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [trackedVideoUrl, setTrackedVideoUrl] = useState(null);
  const [goldenVideoUrl, setGoldenVideoUrl] = useState(null);
  const [goldenPlaybackFrame, setGoldenPlaybackFrame] = useState(0);
  const [rebuildingPreview, setRebuildingPreview] = useState(false);
  const [trackProgress, setTrackProgress] = useState(0);
  const [trackMessage, setTrackMessage] = useState("");
  const [isTracking, setIsTracking] = useState(false);
  const [progress, setProgress] = useState({
    processed: null,
    total: null,
    percent: 0,
    fps: null,
    lastChunkSeedIdx: null,
    goldenMaxIdx: null,
  });

  // Load initial progress
  React.useEffect(() => {
    if (runId) {
      loadProgress();
    }
  }, [runId]);

  const loadProgress = async () => {
    try {
      const prog = await getProgress(runId);
      setProgress({
        processed: prog.golden_processed,
        total: prog.total_frames,
        percent: prog.golden_percent || 0,
        fps: prog.fps || null,
        lastChunkSeedIdx: prog.last_chunk_seed_idx || null,
        goldenMaxIdx: prog.golden_max_idx !== null && prog.golden_max_idx !== undefined ? prog.golden_max_idx : null,
      });
      if (prog.golden_processed > 0) {
        setGoldenVideoUrl(getGoldenVideoUrl(runId) + `?t=${Date.now()}`);
      }
      if (onProgressUpdate) {
        onProgressUpdate(prog);
      }
    } catch (e) {
      console.error("Failed to load progress:", e);
    }
  };

  const handleTrack = async () => {
    setBusy(true);
    setIsTracking(true);
    setError("");
    setSuccess("");
    // Initialize progress immediately so it shows right away
    setTrackProgress(0);
    setTrackMessage("Preparing to track...");

    try {
      const resetVal = autoResetInterval.trim() !== "" ? parseInt(autoResetInterval, 10) : null;
      if (resetVal !== null && (isNaN(resetVal) || resetVal < 1)) {
        throw new Error("Auto-reset interval must be a positive number");
      }

      await trackFrames(runId, nFrames, resetVal, (progress, message, stage) => {
        setTrackProgress(progress);
        setTrackMessage(message || "");
      });
      await loadProgress();
      setTimeout(() => {
        setTrackedVideoUrl(getTrackedVideoUrl(runId) + `?t=${Date.now()}`);
      }, 500);
      setTrackProgress(100);
      setTrackMessage("Completed");
      // Clear progress after a moment
      setTimeout(() => {
        setTrackProgress(0);
        setTrackMessage("");
        setIsTracking(false);
      }, 2000);
    } catch (e) {
      setError(`Failed to track: ${e.message}`);
      setTrackProgress(0);
      setTrackMessage("");
      setIsTracking(false);
    } finally {
      setBusy(false);
      setIsTracking(false);
    }
  };

  const handleCommit = async () => {
    setBusy(true);
    setIsTracking(false); // Not tracking during commit
    setError("");
    setSuccess("");
    // Clear tracking progress during commit (commit doesn't use tracking progress)
    setTrackProgress(0);
    setTrackMessage("");

    try {
      await commitFrames(runId);
      await loadProgress();
      setGoldenVideoUrl(getGoldenVideoUrl(runId) + `?t=${Date.now()}`);
      setSuccess(`✅ Committed! Progress: ${progress.processed}/${progress.total} frames`);
    } catch (e) {
      setError(`Failed to commit: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const refreshGoldenVideo = () => {
    if (runId) {
      setGoldenVideoUrl(getGoldenVideoUrl(runId) + `?t=${Date.now()}`);
    }
  };

  const handleBehaviorLabelUpdated = async () => {
    setRebuildingPreview(true);
    try {
      refreshGoldenVideo();
    } finally {
      setTimeout(() => setRebuildingPreview(false), 800);
    }
  };

  const handleRebuildGoldenPreview = async () => {
    if (!runId) return;
    setRebuildingPreview(true);
    setError("");
    try {
      await rebuildGoldenPreview(runId);
      refreshGoldenVideo();
      setSuccess("Golden preview updated with behaviour labels.");
    } catch (e) {
      setError(`Failed to rebuild golden preview: ${e.message}`);
    } finally {
      setRebuildingPreview(false);
    }
  };

  const behaviorEditorProps = {
    runId,
    defaultFrame: progress.goldenMaxIdx ?? 0,
    maxFrame: progress.goldenMaxIdx,
    videoFrame: goldenPlaybackFrame,
    onLabelUpdated: handleBehaviorLabelUpdated,
    rebuildingPreview,
  };

  const handleDownloadGolden = async () => {
    setBusy(true);
    setError("");
    setSuccess("");

    try {
      await downloadGolden(runId);
      setSuccess("✅ Golden download started (check your browser downloads).");
    } catch (e) {
      setError(`Failed to download golden folder: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const tabs = [
    { id: "tracking", label: "📹 Tracking", icon: "📹" },
    { id: "golden", label: "⭐ Golden", icon: "⭐" },
    { id: "init", label: "🔍 Initialization", icon: "🔍" },
  ];

  return (
    <div style={{ maxWidth: 1400, margin: "0 auto", padding: 24 }}>
      <h1 style={{ marginTop: 0, marginBottom: 32 }}>VOS Annotation App</h1>

      {/* Run info header */}
      <div
        style={{
          padding: 16,
          backgroundColor: "#e7f3ff",
          borderRadius: 8,
          marginBottom: 24,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <strong>Run ID:</strong> <code>{runId}</code>
          {isBehaviorMode && (
            <span
              style={{
                marginLeft: 12,
                padding: "2px 10px",
                borderRadius: 999,
                backgroundColor: "#fff3cd",
                color: "#856404",
                fontSize: 12,
                fontWeight: 600,
              }}
            >
              Behaviour mode
            </span>
          )}
        </div>
        <div>
          <strong>FPS:</strong> {progress.fps || "N/A"} | <strong>Total Frames:</strong> {progress.total || "N/A"}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ marginBottom: 24, borderBottom: "2px solid #dee2e6" }}>
        <div style={{ display: "flex", gap: 8 }}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: "12px 24px",
                fontSize: 14,
                fontWeight: 600,
                backgroundColor: activeTab === tab.id ? "#007bff" : "transparent",
                color: activeTab === tab.id ? "white" : "#495057",
                border: "none",
                borderBottom: activeTab === tab.id ? "3px solid #0056b3" : "3px solid transparent",
                cursor: "pointer",
                borderRadius: "8px 8px 0 0",
                transition: "all 0.2s",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Error/Success messages */}
      {error && (
        <div
          style={{
            padding: 12,
            marginBottom: 16,
            backgroundColor: "#f8d7da",
            border: "1px solid #f5c6cb",
            borderRadius: 6,
            color: "#721c24",
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
            backgroundColor: "#d4edda",
            border: "1px solid #c3e6cb",
            borderRadius: 6,
            color: "#155724",
          }}
        >
          {success}
        </div>
      )}

      {/* Tab content */}
      <div
        style={{
          backgroundColor: "#fff",
          border: "1px solid #dee2e6",
          borderRadius: 12,
          padding: 32,
          boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
        }}
      >
        {activeTab === "tracking" && (
          <div>
            <h2 style={{ marginTop: 0, marginBottom: 24 }}>Tracking</h2>

            {/* Progress */}
            <div style={{ marginBottom: 24, padding: 16, backgroundColor: "#f8f9fa", borderRadius: 8 }}>
              <ProgressDisplay processed={progress.processed} total={progress.total} percent={progress.percent} />
            </div>

            {/* Tracking controls */}
            <div style={{ marginBottom: 24, padding: 16, backgroundColor: "#f8f9fa", borderRadius: 8 }}>
              <h3 style={{ marginTop: 0 }}>Track New Frames</h3>
              
              <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap", marginBottom: 16 }}>
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

            {/* Video previews */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
              <div>
                <VideoPlayer 
                  videoUrl={trackedVideoUrl} 
                  label="Tracked chunk preview (current)" 
                  height={480}
                  progress={isTracking ? trackProgress : null}
                  progressMessage={isTracking ? trackMessage : null}
                />
              </div>
              <div>
                <VideoPlayer
                  videoUrl={goldenVideoUrl}
                  label="Golden preview (committed so far)"
                  height={480}
                  fps={progress.fps}
                  onPlaybackFrame={isBehaviorMode ? setGoldenPlaybackFrame : null}
                />
              </div>
            </div>

            {isBehaviorMode && runId && goldenVideoUrl && (
              <>
                <BehaviorEditor {...behaviorEditorProps} />
                <button
                  type="button"
                  onClick={handleRebuildGoldenPreview}
                  disabled={busy || rebuildingPreview}
                  style={{
                    marginTop: 8,
                    padding: "8px 14px",
                    fontSize: 13,
                    fontWeight: 600,
                    border: "1px solid #6c757d",
                    borderRadius: 8,
                    background: "#fff",
                    cursor: busy || rebuildingPreview ? "not-allowed" : "pointer",
                  }}
                >
                  {rebuildingPreview ? "Rebuilding preview…" : "Rebuild golden preview (refresh labels on video)"}
                </button>
              </>
            )}

            {/* Corrections section */}
            <div style={{ marginTop: 48, paddingTop: 32, borderTop: "2px solid #dee2e6" }}>
              <h2 style={{ marginTop: 0, marginBottom: 24 }}>Corrections</h2>
              {runId && progress.fps !== null && (
                <CorrectionWorkflow
                  runId={runId}
                  trackedVideoUrl={trackedVideoUrl}
                  fps={progress.fps}
                  seedIdx={
                    progress.lastChunkSeedIdx !== null
                      ? progress.lastChunkSeedIdx
                      : progress.goldenMaxIdx !== null
                        ? progress.goldenMaxIdx
                        : 0
                  }
                  onCorrectionApplied={loadProgress}
                />
              )}
            </div>
          </div>
        )}

        {activeTab === "golden" && (
          <div>
            <h2 style={{ marginTop: 0, marginBottom: 24 }}>Golden Preview</h2>
            <div style={{ marginBottom: 24 }}>
              <VideoPlayer
                videoUrl={goldenVideoUrl}
                label="Golden video (all committed frames)"
                height={600}
                fps={progress.fps}
                onPlaybackFrame={isBehaviorMode ? setGoldenPlaybackFrame : null}
              />
            </div>
            {isBehaviorMode && runId && (
              <>
                <BehaviorEditor {...behaviorEditorProps} />
                <button
                  type="button"
                  onClick={handleRebuildGoldenPreview}
                  disabled={busy || rebuildingPreview}
                  style={{
                    marginTop: 8,
                    marginBottom: 16,
                    padding: "8px 14px",
                    fontSize: 13,
                    fontWeight: 600,
                    border: "1px solid #6c757d",
                    borderRadius: 8,
                    background: "#fff",
                    cursor: busy || rebuildingPreview ? "not-allowed" : "pointer",
                  }}
                >
                  {rebuildingPreview ? "Rebuilding preview…" : "Rebuild golden preview (refresh labels on video)"}
                </button>
              </>
            )}

            <button
              onClick={handleDownloadGolden}
              disabled={busy || !runId || !goldenVideoUrl}
              style={{
                marginTop: 24,
                padding: "12px 24px",
                fontSize: 16,
                fontWeight: 600,
                backgroundColor: busy || !runId || !goldenVideoUrl ? "#ccc" : "#17a2b8",
                color: "white",
                border: "none",
                borderRadius: 6,
                cursor: busy || !runId || !goldenVideoUrl ? "not-allowed" : "pointer",
                width: "100%",
              }}
            >
              {busy ? "Downloading..." : "💾 Download golden folder"}
            </button>
          </div>
        )}

        {activeTab === "init" && (
          <div>
            <h2 style={{ marginTop: 0, marginBottom: 24 }}>Initialization Preview</h2>
            {frame0Image && <Frame0Preview imageDataUrl={frame0Image} />}
          </div>
        )}
      </div>
    </div>
  );
}
