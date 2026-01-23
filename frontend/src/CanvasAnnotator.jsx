import React, { useEffect, useMemo, useRef } from "react";
import { maskPngFromB64 } from "./api.js";

function getMousePosOnCanvas(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  return { x, y };
}

export default function CanvasAnnotator({
  imageUrl,
  width,
  height,
  selectedMaskB64,
  points,
  onAddPoint,
}) {
  const baseRef = useRef(null);
  const pointsRef = useRef(null);

  const maskUrl = useMemo(() => {
    if (!selectedMaskB64) return null;
    return maskPngFromB64(selectedMaskB64);
  }, [selectedMaskB64]);

  // Draw base image + mask overlay
  useEffect(() => {
    const base = baseRef.current;
    const pts = pointsRef.current;
    if (!base || !pts || !imageUrl) return;

    base.width = width;
    base.height = height;
    pts.width = width;
    pts.height = height;

    const ctx = base.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;

    img.onload = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0, width, height);

      if (!maskUrl) return;

      const m = new Image();
      m.crossOrigin = "anonymous";
      m.src = maskUrl;

      m.onload = () => {
        // mask -> red transparent overlay
        const off = document.createElement("canvas");
        off.width = width;
        off.height = height;
        const octx = off.getContext("2d");

        octx.drawImage(m, 0, 0, width, height);
        const imgData = octx.getImageData(0, 0, width, height);
        const data = imgData.data;

        for (let i = 0; i < data.length; i += 4) {
          const v = data[i]; // grayscale
          if (v > 127) {
            data[i] = 255;     // R
            data[i + 1] = 0;   // G
            data[i + 2] = 0;   // B
            data[i + 3] = 90;  // alpha
          } else {
            data[i + 3] = 0;
          }
        }

        octx.putImageData(imgData, 0, 0);
        ctx.drawImage(off, 0, 0);
      };
    };

    return () => {
      if (maskUrl) URL.revokeObjectURL(maskUrl);
    };
  }, [imageUrl, maskUrl, width, height]);

  // Draw points (always on top, fast)
  useEffect(() => {
    const pts = pointsRef.current;
    if (!pts) return;

    pts.width = width;
    pts.height = height;

    const ctx = pts.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    for (const p of points) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);

      // +1 = green, -1 = red
      ctx.fillStyle = p.label === 1 ? "green" : "red";

      // outline for visibility
      ctx.strokeStyle = "black";
      ctx.lineWidth = 2;

      ctx.fill();
      ctx.stroke();
    }
  }, [points, width, height]);

  function handleMouseDown(e) {
    e.preventDefault();
    const pts = pointsRef.current;
    if (!pts) return;

    // left = positive, right = negative
    const label = e.button === 2 ? -1 : 1;
    const { x, y } = getMousePosOnCanvas(e, pts);

    onAddPoint({ x, y, label });
  }

  return (
    <div style={{ display: "inline-block" }}>
      <div
        style={{
          position: "relative",
          width: `${width}px`,
          height: `${height}px`,
          border: "1px solid #333",
          userSelect: "none",
        }}
      >
        <canvas
          ref={baseRef}
          style={{ position: "absolute", left: 0, top: 0 }}
        />
        <canvas
          ref={pointsRef}
          onMouseDown={handleMouseDown}
          onContextMenu={(e) => e.preventDefault()}
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            cursor: "crosshair",
          }}
        />
      </div>

      <div style={{ padding: 8, fontSize: 12, color: "#666" }}>
        Left click = <span style={{ color: "green" }}>positive</span> • Right click ={" "}
        <span style={{ color: "red" }}>negative</span>
      </div>
    </div>
  );
}
