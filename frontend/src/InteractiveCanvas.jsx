import React, { useEffect, useRef, useState } from "react";

/**
 * Interactive canvas for adding positive/negative points to refine masks
 * @param {string} imageDataUrl - Base64 data URL of the image
 * @param {number} width - Canvas width
 * @param {number} height - Canvas height
 * @param {Array} points - Array of {x, y, is_positive} points
 * @param {Function} onAddPoint - Callback when point is added: (x, y, is_positive) => void
 * @param {Function} onDeletePoint - Optional callback to delete a point: (pointIndex) => void
 * @param {Function} onClearPoints - Optional callback to clear points
 */
export default function InteractiveCanvas({
  imageDataUrl,
  width,
  height,
  points = [],
  onAddPoint,
  onDeletePoint,
  onClearPoints,
}) {
  const canvasRef = useRef(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  // Draw image and points
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageDataUrl) return;

    const ctx = canvas.getContext("2d");
    canvas.width = width;
    canvas.height = height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Load and draw image
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0, width, height);
      setImageLoaded(true);

      // Draw points on top
      drawPoints(ctx);
    };
    img.src = imageDataUrl;
  }, [imageDataUrl, width, height]);

  // Redraw points when they change
  useEffect(() => {
    if (!imageLoaded) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    // Redraw image first
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0, width, height);
      drawPoints(ctx);
    };
    img.src = imageDataUrl;
  }, [points, imageLoaded, imageDataUrl, width, height]);

  const drawPoints = (ctx) => {
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);

      // Green for positive, red for negative
      ctx.fillStyle = p.is_positive ? "#00ff00" : "#ff0000";
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 2;

      ctx.fill();
      ctx.stroke();
    }
  };

  const getMousePos = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  // Check if click is near an existing point (within threshold pixels)
  const findPointNearClick = (clickX, clickY, threshold = 15) => {
    for (let i = points.length - 1; i >= 0; i--) {
      const p = points[i];
      const dx = clickX - p.x;
      const dy = clickY - p.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance <= threshold) {
        return i; // Return index of point to delete
      }
    }
    return null;
  };

  const handleMouseDown = (e) => {
    e.preventDefault();
    
    // Only handle left clicks here (right clicks are handled by contextmenu)
    if (e.button !== 0) return;
    
    const pos = getMousePos(e);
    if (!pos) return;

    // Check if clicking on an existing point (delete it)
    if (onDeletePoint) {
      const pointIndex = findPointNearClick(pos.x, pos.y);
      if (pointIndex !== null) {
        onDeletePoint(pointIndex);
        return;
      }
    }

    // Otherwise, add a new positive point
    onAddPoint(Math.round(pos.x), Math.round(pos.y), true);
  };

  const handleContextMenu = (e) => {
    e.preventDefault(); // Prevent right-click menu
    const pos = getMousePos(e);
    if (!pos) return;

    // Check if clicking on an existing point (delete it)
    if (onDeletePoint) {
      const pointIndex = findPointNearClick(pos.x, pos.y);
      if (pointIndex !== null) {
        onDeletePoint(pointIndex);
        return;
      }
    }

    // Otherwise, add a new negative point
    onAddPoint(Math.round(pos.x), Math.round(pos.y), false);
  };

  return (
    <div style={{ display: "inline-block" }}>
      <div
        style={{
          position: "relative",
          width: `${width}px`,
          height: `${height}px`,
          border: "1px solid #333",
          userSelect: "none",
          cursor: "crosshair",
        }}
      >
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onContextMenu={handleContextMenu}
          style={{
            display: "block",
            width: "100%",
            height: "100%",
          }}
        />
      </div>
      <div style={{ padding: 8, fontSize: 12, color: "#666", backgroundColor: "#f9f9f9" }}>
        <span style={{ color: "green", fontWeight: 500 }}>Left click</span> = add to mask (positive) •{" "}
        <span style={{ color: "red", fontWeight: 500 }}>Right click</span> = remove from mask (negative)
        {onDeletePoint && points.length > 0 && (
          <> • <span style={{ color: "#666", fontWeight: 500 }}>Click on point</span> = delete</>
        )}
        {points.length > 0 && (
          <>
            {" "}• {points.length} point{points.length !== 1 ? "s" : ""} added
            {onClearPoints && (
              <button
                onClick={onClearPoints}
                style={{
                  marginLeft: 8,
                  padding: "2px 8px",
                  fontSize: 11,
                  backgroundColor: "#ff6b6b",
                  color: "white",
                  border: "none",
                  borderRadius: 3,
                  cursor: "pointer",
                }}
              >
                Clear all
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
