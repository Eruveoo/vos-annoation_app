import React, { useState, useEffect } from "react";
import VideoSelectionPage from "./pages/VideoSelectionPage.jsx";
import IDAssignmentPage from "./pages/IDAssignmentPage.jsx";
import MainWorkspacePage from "./pages/MainWorkspacePage.jsx";
import { testConnection, applyInitIds, getProgress } from "./api.js";
import { ANNOTATION_MODES } from "./behaviorLabels.js";

export default function App() {
  // Page navigation state
  const [currentPage, setCurrentPage] = useState("video-selection"); // video-selection, id-assignment, main-workspace

  // Connection status
  const [connectionStatus, setConnectionStatus] = useState(null);

  // Video selection state
  const [videoPath, setVideoPath] = useState("video_sample_5_min.mp4");

  // Annotation mode: standard VOS vs behaviour labelling
  const [annotationMode, setAnnotationMode] = useState(ANNOTATION_MODES.STANDARD);

  // Initialize state
  const [runId, setRunId] = useState(null);
  const [frame0Image, setFrame0Image] = useState(null);
  const [maskAssignments, setMaskAssignments] = useState([]);

  // Resume info for VideoSelectionPage (optional)
  const [resumeRunId, setResumeRunId] = useState("");

  // ID assignment state
  const [idMapping, setIdMapping] = useState({});

  // Progress state (for main workspace)
  const [progress, setProgress] = useState({
    processed: null,
    total: null,
    percent: 0,
    fps: null,
    lastChunkSeedIdx: null,
    goldenMaxIdx: null,
  });

  // Test backend connection on mount
  useEffect(() => {
    testConnection().then((connected) => {
      setConnectionStatus(connected);
    });
  }, []);

  // Load last run id (best-effort)
  useEffect(() => {
    try {
      const last = window.localStorage.getItem("vos_last_run_id");
      if (last && typeof last === "string") {
        setResumeRunId(last);
      }
    } catch {
      // ignore
    }
  }, []);

  // Handle video prepared - go to initialize page
  const handleVideoLoaded = ({ videoPath: path, runId: preparedRunId, annotationMode: mode }) => {
    setVideoPath(path);
    setRunId(preparedRunId);
    if (mode) {
      setAnnotationMode(mode);
    }
    setCurrentPage("video-selection");
  };

  const handleAnnotationModeChange = (mode) => {
    setAnnotationMode(mode);
  };

  // Handle initialized - go to ID assignment page
  const handleInitialized = (result) => {
    setRunId(result.run_id);
    setFrame0Image(result.image);
    setMaskAssignments(result.mask_assignments);
    
    // Initialize mapping with auto-assigned IDs
    const initialMapping = {};
    result.mask_assignments.forEach((assignment) => {
      initialMapping[assignment.mask_index] = assignment.auto_assigned_id;
    });
    setIdMapping(initialMapping);
    
    setCurrentPage("id-assignment");
  };

  // Handle IDs applied - go to main workspace
  const handleIdsApplied = async (mapping, behaviorByCowId = null) => {
    if (!runId) {
      return;
    }

    // Filter out deleted masks (undefined values)
    const validMapping = {};
    Object.entries(mapping).forEach(([maskIndex, finalId]) => {
      if (finalId !== undefined && finalId >= 1) {
        validMapping[maskIndex] = finalId;
      }
    });

    if (Object.keys(validMapping).length === 0) {
      alert("No valid IDs assigned. Please assign at least one ID.");
      return;
    }

    try {
      await applyInitIds(runId, validMapping, behaviorByCowId);
      
      // Refresh progress after initialization
      const prog = await getProgress(runId);
      setProgress({
        processed: prog.golden_processed,
        total: prog.total_frames,
        percent: prog.golden_percent || 0,
        fps: prog.fps || null,
        lastChunkSeedIdx: prog.last_chunk_seed_idx || null,
        goldenMaxIdx: prog.golden_max_idx !== null && prog.golden_max_idx !== undefined ? prog.golden_max_idx : null,
      });
      if (prog.annotation_mode) {
        setAnnotationMode(prog.annotation_mode);
      }
      
      // Go to main workspace
      setCurrentPage("main-workspace");
    } catch (e) {
      alert(`Failed to apply IDs: ${e.message}`);
    }
  };

  const handleResumeSession = async (inputRunId) => {
    const rid = (inputRunId || "").trim();
    if (!rid) return;
    setRunId(rid);
    try {
      window.localStorage.setItem("vos_last_run_id", rid);
    } catch {
      // ignore
    }

    try {
      const prog = await getProgress(rid);
      if (prog.annotation_mode) {
        setAnnotationMode(prog.annotation_mode);
      }
      if (prog.golden_max_idx !== null && prog.golden_max_idx !== undefined) {
        setCurrentPage("main-workspace");
      } else {
        setCurrentPage("video-selection");
      }
    } catch {
      setCurrentPage("video-selection");
    }
  };

  // Render current page
  switch (currentPage) {
    case "video-selection":
      return (
        <VideoSelectionPage
          onVideoLoaded={handleVideoLoaded}
          onInitialized={handleInitialized}
          onResumeSession={handleResumeSession}
          connectionStatus={connectionStatus}
          defaultResumeRunId={resumeRunId}
          annotationMode={annotationMode}
          onAnnotationModeChange={handleAnnotationModeChange}
        />
      );

    case "id-assignment":
      return (
        <IDAssignmentPage
          runId={runId}
          frame0Image={frame0Image}
          maskAssignments={maskAssignments}
          annotationMode={annotationMode}
          onIdsApplied={handleIdsApplied}
          onBack={() => setCurrentPage("video-selection")}
        />
      );

    case "main-workspace":
      return (
        <MainWorkspacePage
          runId={runId}
          frame0Image={frame0Image}
          annotationMode={annotationMode}
          onProgressUpdate={(prog) => {
            setProgress({
              processed: prog.golden_processed,
              total: prog.total_frames,
              percent: prog.golden_percent || 0,
              fps: prog.fps || null,
              lastChunkSeedIdx: prog.last_chunk_seed_idx || null,
              goldenMaxIdx: prog.golden_max_idx !== null && prog.golden_max_idx !== undefined ? prog.golden_max_idx : null,
            });
            if (prog.annotation_mode) {
              setAnnotationMode(prog.annotation_mode);
            }
          }}
        />
      );

    default:
      return <div>Unknown page: {currentPage}</div>;
  }
}
