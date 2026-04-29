import React, { useState } from "react";
import Frame0Preview from "../Frame0Preview.jsx";
import { initSam } from "../api.js";

export default function InitializePage({ videoPath, runId, onInitialized, onBack }) {
  const [prompt, setPrompt] = useState("cow");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const handleInitialize = async () => {
    if (!prompt.trim()) {
      setError("Please enter a text prompt");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const initResult = await initSam(runId, prompt.trim());
      setResult(initResult);
    } catch (e) {
      setError(`Failed to initialize: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: 24 }}>
      <h1 style={{ marginTop: 0, marginBottom: 32 }}>VOS Annotation App</h1>

      <div
        style={{
          backgroundColor: "#fff",
          border: "1px solid #dee2e6",
          borderRadius: 12,
          padding: 32,
          boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
        }}
      >
        <div style={{ marginBottom: 24, display: "flex", alignItems: "center", gap: 16 }}>
          <button
            onClick={onBack}
            style={{
              padding: "8px 16px",
              fontSize: 14,
              backgroundColor: "#6c757d",
              color: "white",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
            }}
          >
            ← Back
          </button>
          <h2 style={{ margin: 0 }}>2. Initialize (SAM on Frame 0)</h2>
        </div>

        <div style={{ marginBottom: 16, padding: 12, backgroundColor: "#e7f3ff", borderRadius: 8, fontSize: 14 }}>
          <div><strong>Video:</strong> {videoPath}</div>
          <div><strong>Run ID:</strong> <code>{runId}</code></div>
        </div>

        {/* Prompt input */}
        <div style={{ marginBottom: 24 }}>
          <label
            style={{
              display: "block",
              marginBottom: 8,
              fontSize: 14,
              fontWeight: 600,
              color: "#495057",
            }}
          >
            Text prompt for SAM:
          </label>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="cow"
            disabled={loading}
            style={{
              width: "100%",
              padding: 12,
              border: "1px solid #ced4da",
              borderRadius: 6,
              fontSize: 14,
              boxSizing: "border-box",
            }}
          />
          <div style={{ fontSize: 12, color: "#6c757d", marginTop: 4 }}>
            Enter a text description of the objects you want to segment (e.g., "cow", "person", "car")
          </div>
        </div>

        {/* Initialize button */}
        <button
          onClick={handleInitialize}
          disabled={loading || !prompt.trim()}
          style={{
            padding: "12px 24px",
            fontSize: 16,
            fontWeight: 600,
            backgroundColor: loading || !prompt.trim() ? "#6c757d" : "#007bff",
            color: "white",
            border: "none",
            borderRadius: 6,
            cursor: loading || !prompt.trim() ? "not-allowed" : "pointer",
            width: "100%",
            marginBottom: 24,
          }}
        >
          {loading ? "⏳ Running SAM on frame 0..." : "🔍 Run SAM on Frame 0"}
        </button>

        {/* Error message */}
        {error && (
          <div
            style={{
              padding: 12,
              marginBottom: 24,
              backgroundColor: "#f8d7da",
              border: "1px solid #f5c6cb",
              borderRadius: 6,
              color: "#721c24",
            }}
          >
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div style={{ marginTop: 32 }}>
            <div
              style={{
                padding: 16,
                backgroundColor: "#d4edda",
                border: "1px solid #c3e6cb",
                borderRadius: 8,
                marginBottom: 24,
              }}
            >
              <strong>✅ Initialization complete!</strong> Found {result.mask_assignments.length} masks.
            </div>

            {/* Preview */}
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ marginBottom: 16 }}>Frame 0 Preview</h3>
              <Frame0Preview imageDataUrl={result.image} />
            </div>

            {/* Proceed button */}
            <button
              onClick={() => onInitialized(result)}
              style={{
                padding: "12px 24px",
                fontSize: 16,
                fontWeight: 600,
                backgroundColor: "#28a745",
                color: "white",
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
                width: "100%",
              }}
            >
              Next: Assign IDs →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
