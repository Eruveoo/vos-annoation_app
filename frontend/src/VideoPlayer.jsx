import React, { useState, useRef, useEffect, useImperativeHandle, forwardRef } from "react";

/**
 * Video player component
 * @param {string} videoUrl - URL to the video
 * @param {string} label - Label for the video
 * @param {number} height - Display height
 * @param {number} progress - Optional progress percentage (0-100)
 * @param {string} progressMessage - Optional progress message
 */
const VideoPlayer = forwardRef(function VideoPlayer(
  {
    videoUrl,
    label,
    height = 480,
    progress = null,
    progressMessage = null,
    fps = null,
    onPlaybackFrame = null,
  },
  ref
) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);

  useImperativeHandle(ref, () => ({
    seekToFrame(frame) {
      const video = videoRef.current;
      if (!video || !fps) return;
      const t = Math.max(0, Number(frame) / fps);
      video.currentTime = t;
      if (onPlaybackFrame) {
        onPlaybackFrame(Math.max(0, Math.floor(t * fps)));
      }
    },
  }));

  useEffect(() => {
    if (videoUrl && videoRef.current) {
      setLoading(true);
      setError(null);
    }
  }, [videoUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !fps || !onPlaybackFrame) return;

    const handleTimeUpdate = () => {
      const frameIdx = Math.max(0, Math.floor(video.currentTime * fps));
      onPlaybackFrame(frameIdx);
    };

    video.addEventListener("timeupdate", handleTimeUpdate);
    video.addEventListener("seeked", handleTimeUpdate);
    return () => {
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("seeked", handleTimeUpdate);
    };
  }, [videoUrl, fps, onPlaybackFrame]);

  const handleCanPlay = () => {
    setLoading(false);
    setError(null);
  };

  const handleError = (e) => {
    setLoading(false);
    const video = e.target;
    let errorMsg = "Failed to load video.";
    
    if (video.error) {
      switch (video.error.code) {
        case video.error.MEDIA_ERR_ABORTED:
          errorMsg = "Video loading was aborted.";
          break;
        case video.error.MEDIA_ERR_NETWORK:
          errorMsg = "Network error while loading video. Check your connection.";
          break;
        case video.error.MEDIA_ERR_DECODE:
          errorMsg = "Video decoding error. The video codec may not be supported by your browser.";
          break;
        case video.error.MEDIA_ERR_SRC_NOT_SUPPORTED:
          errorMsg = "Video format not supported. The video file may need to be re-encoded.";
          break;
        default:
          errorMsg = `Video error (code: ${video.error.code}). The video file may not be ready or the format is not supported.`;
      }
    } else {
      // No error code - might be a network issue or file not found
      errorMsg = "Video failed to load. The file may not exist or there's a network issue.";
    }
    
    setError(errorMsg);
    console.error("Video error:", {
      code: video.error?.code,
      message: video.error?.message,
      src: videoUrl,
      readyState: video.readyState,
      networkState: video.networkState,
    });
  };

  if (!videoUrl) {
    return (
      <div>
        {label && (
          <div style={{ marginBottom: 8, fontWeight: 500, fontSize: 14 }}>
            {label}
          </div>
        )}
        <div
          style={{
            width: "100%",
            height: `${height}px`,
            border: "1px solid #333",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            color: "#666",
            backgroundColor: "#000",
            padding: "40px 40px",
            boxSizing: "border-box",
          }}
        >
          {progress !== null && progress > 0 ? (
            <div style={{ 
              width: "100%", 
              maxWidth: 400,
            }}>
              <div style={{ 
                width: "100%", 
                height: 8, 
                backgroundColor: "#e9ecef", 
                borderRadius: 4,
                overflow: "hidden",
                marginBottom: 12
              }}>
                <div style={{
                  height: "100%",
                  width: `${progress}%`,
                  backgroundColor: "#28a745",
                  borderRadius: 4,
                  transition: "width 0.3s ease"
                }} />
              </div>
              {progressMessage && (
                <div style={{ fontSize: 14, color: "#6c757d", textAlign: "center" }}>
                  {progressMessage}
                </div>
              )}
            </div>
          ) : (
            <div style={{ color: "#666" }}>
              No video available
            </div>
          )}
        </div>
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
        {/* Progress bar overlay - shows when progress is provided, even if video exists */}
        {progress !== null && progress >= 0 && (
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              backgroundColor: "rgba(0, 0, 0, 0.9)",
              zIndex: 20,
              padding: 40,
            }}
          >
            <div style={{ 
              width: "100%", 
              maxWidth: 400,
            }}>
              <div style={{ 
                width: "100%", 
                height: 8, 
                backgroundColor: "#e9ecef", 
                borderRadius: 4,
                overflow: "hidden",
                marginBottom: 12
              }}>
                <div style={{
                  height: "100%",
                  width: `${progress}%`,
                  backgroundColor: "#28a745",
                  borderRadius: 4,
                  transition: "width 0.3s ease"
                }} />
              </div>
              {progressMessage && (
                <div style={{ fontSize: 14, color: "#6c757d", textAlign: "center" }}>
                  {progressMessage}
                </div>
              )}
            </div>
          </div>
        )}
        {loading && (
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              backgroundColor: "rgba(0, 0, 0, 0.7)",
              color: "white",
              zIndex: 10,
            }}
          >
            Loading video...
          </div>
        )}
        {error && (
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              backgroundColor: "rgba(255, 0, 0, 0.1)",
              color: "#c00",
              zIndex: 10,
              padding: 16,
              textAlign: "center",
            }}
          >
            {error}
          </div>
        )}
        <video
          ref={videoRef}
          src={videoUrl}
          controls
          preload="metadata"
          onCanPlay={handleCanPlay}
          onError={handleError}
          onLoadStart={() => setLoading(true)}
          onLoadedMetadata={() => {
            // Video metadata loaded successfully
            setLoading(false);
            setError(null);
          }}
          onWaiting={() => setLoading(true)}
          onPlaying={() => setLoading(false)}
          style={{
            width: "100%",
            height: `${height}px`,
            border: "1px solid #333",
            backgroundColor: "#000",
          }}
        >
          Your browser does not support the video tag.
        </video>
      </div>
    </div>
  );
});

export default VideoPlayer;
