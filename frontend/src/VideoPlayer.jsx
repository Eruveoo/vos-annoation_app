import React, { useState, useRef, useEffect } from "react";

/**
 * Video player component
 * @param {string} videoUrl - URL to the video
 * @param {string} label - Label for the video
 * @param {number} height - Display height
 */
export default function VideoPlayer({ videoUrl, label, height = 480 }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);

  useEffect(() => {
    if (videoUrl && videoRef.current) {
      setLoading(true);
      setError(null);
    }
  }, [videoUrl]);

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
}
