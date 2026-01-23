import React, { useMemo, useState } from "react";
import CanvasAnnotator from "./CanvasAnnotator.jsx";
import { initSegmentation, refineMask } from "./api.js";

export default function App() {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);

  const [imageId, setImageId] = useState(null);
  const [imgW, setImgW] = useState(0);
  const [imgH, setImgH] = useState(0);

  const [instances, setInstances] = useState([]);
  const [selectedInstanceId, setSelectedInstanceId] = useState(null);

  const [points, setPoints] = useState([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const selected = useMemo(() => {
    if (selectedInstanceId == null) return null;
    return instances.find((x) => x.instance_id === selectedInstanceId) || null;
  }, [instances, selectedInstanceId]);

  function onPickFile(f) {
    setFile(f);
    setErr("");
    setImageId(null);
    setInstances([]);
    setSelectedInstanceId(null);
    setPoints([]);

    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setImageUrl(URL.createObjectURL(f));
  }

  async function runInit() {
    if (!file) return;
    setBusy(true);
    setErr("");
    try {
      const res = await initSegmentation(file, "cow");
      setImageId(res.image_id);
      setImgW(res.width);
      setImgH(res.height);
      setInstances(res.instances);
      setSelectedInstanceId(res.instances?.[0]?.instance_id ?? null);
      setPoints([]);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function runRefine() {
    if (!imageId || !selectedInstanceId) return;
    if (points.length === 0) return;

    setBusy(true);
    setErr("");
    try {
      const res = await refineMask({
        image_id: imageId,
        instance_id: selectedInstanceId,
        points
      });

      // replace mask for selected instance
      setInstances((prev) =>
        prev.map((inst) =>
          inst.instance_id === selectedInstanceId
            ? { ...inst, mask_png_b64: res.mask_png_b64, score: res.score, box: res.box }
            : inst
        )
      );
      setPoints([]);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 16 }}>
      <h2 style={{ marginTop: 0 }}>SAM Cow Annotator (MVP)</h2>

      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
        {/* Left panel */}
        <div style={{ width: 320 }}>
          <div style={{ marginBottom: 12 }}>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => e.target.files?.[0] && onPickFile(e.target.files[0])}
            />
          </div>

          <button disabled={!file || busy} onClick={runInit}>
            {busy ? "Working..." : "Run initial 'cow' segmentation"}
          </button>

          <div style={{ marginTop: 16 }}>
            <h4>Instances</h4>
            {instances.length === 0 ? (
              <div style={{ color: "#666" }}>No instances yet.</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {instances.map((inst) => (
                  <button
                    key={inst.instance_id}
                    onClick={() => {
                      setSelectedInstanceId(inst.instance_id);
                      setPoints([]);
                    }}
                    style={{
                      textAlign: "left",
                      padding: 8,
                      border: "1px solid #ccc",
                      background:
                        inst.instance_id === selectedInstanceId ? "#eee" : "white",
                      cursor: "pointer",
                    }}
                  >
                    <div><b>Cow {inst.instance_id}</b></div>
                    <div style={{ fontSize: 12, color: "#666" }}>
                      score: {Number(inst.score).toFixed(3)}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div style={{ marginTop: 16 }}>
            <h4>Edit</h4>
            <div style={{ fontSize: 12, color: "#666", marginBottom: 8 }}>
              Add points on the image, then refine.
            </div>

            <button disabled={busy || !selected || points.length === 0} onClick={runRefine}>
              Refine selected mask ({points.length} points)
            </button>

            <button
              style={{ marginLeft: 8 }}
              disabled={busy || points.length === 0}
              onClick={() => setPoints([])}
            >
              Clear points
            </button>
          </div>

          {err && (
            <pre style={{ marginTop: 16, color: "crimson", whiteSpace: "pre-wrap" }}>
              {err}
            </pre>
          )}
        </div>

        {/* Right: canvas */}
        <div>
          {imageUrl ? (
            <CanvasAnnotator
              imageUrl={imageUrl}
              width={imgW || 800}
              height={imgH || 600}
              selectedMaskB64={selected?.mask_png_b64 ?? null}
              points={points}
              onAddPoint={(p) => setPoints((prev) => [...prev, p])}
            />
          ) : (
            <div style={{ color: "#666" }}>Upload an image to start.</div>
          )}
        </div>
      </div>
    </div>
  );
}
