import React, { useEffect, useRef } from "react";

/**
 * Component to display frame 0 with SAM masks overlay
 * @param {string} imageDataUrl - Base64 data URL of the preview image
 * @param {number} width - Display width (optional, auto if not provided)
 * @param {number} height - Display height (optional, auto if not provided)
 */
export default function Frame0Preview({ imageDataUrl, width, height }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  useEffect(() => {
    if (!imageDataUrl || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();

    img.onload = () => {
      // Set canvas size to match image
      if (width && height) {
        canvas.width = width;
        canvas.height = height;
      } else {
        canvas.width = img.width;
        canvas.height = img.height;
      }

      // Clear and draw image
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };

    img.src = imageDataUrl;
    imgRef.current = img;
  }, [imageDataUrl, width, height]);

  if (!imageDataUrl) {
    return (
      <div
        style={{
          width: width || 800,
          height: height || 600,
          border: "1px solid #ccc",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#666",
          backgroundColor: "#f5f5f5",
        }}
      >
        No preview available
      </div>
    );
  }

  return (
    <div style={{ display: "inline-block" }}>
      <canvas
        ref={canvasRef}
        style={{
          maxWidth: "100%",
          height: "auto",
          border: "1px solid #333",
          display: "block",
        }}
      />
      <div style={{ marginTop: 8, fontSize: 12, color: "#666" }}>
        Frame 0 with SAM masks (auto-assigned IDs shown)
      </div>
    </div>
  );
}
