import React, { useState, useRef, useEffect } from "react";

/**
 * Interactive video player that reports current frame
 * @param {string} videoUrl - URL to the video
 * @param {string} label - Label for the video
 * @param {number} height - Display height
 * @param {number} fps - Video FPS
 * @param {number} seedIdx - Seed frame index (to calculate absolute frame)
 * @param {Function} onFrameChange - Callback when frame changes (receives absoluteFrameIdx)
 * @param {Function} onSelectFrame - Optional callback when user clicks "Select this frame" button
 * @param {Function} onCorrectFrame - Optional callback when user clicks "Correct current frame" button
 * @param {number} selectedFrame - Currently selected frame (to highlight)
 * @param {boolean} busy - Whether correction is in progress (to disable buttons)
 */
export default function InteractiveVideoPlayer({
  videoUrl,
  label,
  height = 480,
  fps,
  seedIdx,
  onFrameChange,
  onSelectFrame,
  onCorrectFrame,
  selectedFrame,
  busy = false,
}) {
  const [currentFrame, setCurrentFrame] = useState(null);
  const videoRef = useRef(null);
  const updateIntervalRef = useRef(null);

  useEffect(() => {
    if (!videoUrl || !fps || seedIdx === null || !videoRef.current) {
      return;
    }

    const updateFrame = () => {
      const video = videoRef.current;
      if (video && video.readyState >= 2) {
        // readyState 2 = HAVE_CURRENT_DATA
        const currentTime = video.currentTime;
        // Calculate relative frame in the tracked chunk
        // The tracked video starts at seed frame, so frame 0 in video = seed frame
        const relativeFrame = Math.floor(currentTime * fps);
        // Convert to absolute frame
        const absoluteFrame = seedIdx + relativeFrame;
        setCurrentFrame(absoluteFrame);
        if (onFrameChange) {
          onFrameChange(absoluteFrame);
        }
      }
    };

    // Update on timeupdate events
    const video = videoRef.current;
    video.addEventListener("timeupdate", updateFrame);
    video.addEventListener("seeked", updateFrame);

    // Also update periodically for more accuracy
    updateIntervalRef.current = setInterval(updateFrame, 100);

    return () => {
      video.removeEventListener("timeupdate", updateFrame);
      video.removeEventListener("seeked", updateFrame);
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
      }
    };
  }, [videoUrl, fps, seedIdx, onFrameChange]);

  if (!videoUrl) {
    return (
      <div
        style={{
          width: "100%",
          height: `${height}px`,
          border: "1px solid #ccc",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#666",
          backgroundColor: "#f5f5f5",
        }}
      >
        {label ? `${label}: No video available` : "No video available"}
      </div>
    );
  }

  return (
    <div>
      {label && (
        <div style={{ marginBottom: 8, fontWeight: 500, fontSize: 14 }}>
          {label}
        </div>
      )}
      <div style={{ position: "relative" }}>
        <video
          ref={videoRef}
          src={videoUrl}
          controls
          preload="auto"
          style={{
            width: "100%",
            height: `${height}px`,
            border: "1px solid #333",
            backgroundColor: "#000",
          }}
        >
          Your browser does not support the video tag.
        </video>
        {currentFrame !== null && (
          <div
            style={{
              position: "absolute",
              top: 8,
              right: 8,
              backgroundColor:
                selectedFrame === currentFrame
                  ? "rgba(40, 167, 69, 0.9)"
                  : "rgba(0, 0, 0, 0.7)",
              color: "white",
              padding: "4px 8px",
              borderRadius: 4,
              fontSize: 12,
              fontFamily: "monospace",
              fontWeight: selectedFrame === currentFrame ? "bold" : "normal",
            }}
          >
            Frame: {currentFrame}
            {selectedFrame === currentFrame && " ✓"}
          </div>
        )}
      </div>

      {/* Controls BELOW the video so they're never hidden by the video UI */}
      <div
        style={{
          marginTop: 12,
          display: "flex",
          gap: 12,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <div style={{ fontSize: 12, color: "#666", fontFamily: "monospace" }}>
          {currentFrame !== null ? (
            <>
              Current frame: <strong>{currentFrame}</strong>
              {selectedFrame !== null && (
                <>
                  {" "}
                  • Selected: <strong>{selectedFrame}</strong>
                </>
              )}
            </>
          ) : (
            "Current frame: — (press play or scrub to update)"
          )}
        </div>

        {onSelectFrame && (
          <button
            onClick={() => currentFrame !== null && onSelectFrame(currentFrame)}
            disabled={currentFrame === null || busy}
            style={{
              padding: "8px 12px",
              fontSize: 13,
              fontWeight: 600,
              backgroundColor:
                currentFrame !== null && selectedFrame === currentFrame
                  ? "#28a745"
                  : "#ff9800",
              color: "white",
              border: "none",
              borderRadius: 6,
              cursor: currentFrame === null || busy ? "not-allowed" : "pointer",
              opacity: currentFrame === null || busy ? 0.6 : 1,
            }}
          >
            {currentFrame !== null && selectedFrame === currentFrame
              ? "✓ Selected"
              : "Select current frame"}
          </button>
        )}

        {onCorrectFrame && (
          <button
            onClick={() => currentFrame !== null && onCorrectFrame(currentFrame)}
            disabled={currentFrame === null || busy || currentFrame < 1}
            style={{
              padding: "8px 12px",
              fontSize: 13,
              fontWeight: 600,
              backgroundColor: busy ? "#ccc" : "#2196f3",
              color: "white",
              border: "none",
              borderRadius: 6,
              cursor: currentFrame === null || busy || currentFrame < 1 ? "not-allowed" : "pointer",
              opacity: currentFrame === null || busy || currentFrame < 1 ? 0.6 : 1,
            }}
          >
            {busy ? "Preparing..." : "🔧 Correct current frame"}
          </button>
        )}
      </div>
    </div>
  );
}
