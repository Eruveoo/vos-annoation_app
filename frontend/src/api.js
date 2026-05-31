const BACKEND = "http://127.0.0.1:12212";

/**
 * Test backend connection
 */
export async function testConnection() {
  try {
    const res = await fetch(`${BACKEND}/health`, {
      method: "GET",
    });
    return res.ok;
  } catch (e) {
    console.error("[API] Connection test failed:", e);
    return false;
  }
}

/**
 * Initialize a new annotation session
 * @param {string} videoPath - Path to video file on server
 * @param {string} prompt - Text prompt for SAM (e.g., "cow")
 * @returns {Promise<{run_id: string, fps: number, n_frames_total: number, image: string, mask_assignments: Array}>}
 */
export async function initVideo(videoPath, prompt = "cow") {
  const params = new URLSearchParams({ video_path: videoPath, prompt });
  
  // Create an AbortController for timeout handling
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minute timeout (videos can be large)
  
  try {
    console.log(`[API] Fetching ${BACKEND}/init?${params.toString()}`);
    const res = await fetch(`${BACKEND}/init?${params}`, {
      method: "POST",
      signal: controller.signal,
      headers: {
        "Accept": "application/json",
      },
    });
    
    clearTimeout(timeoutId);
    
    console.log(`[API] Response status: ${res.status} ${res.statusText}`);
    console.log(`[API] Response headers:`, Object.fromEntries(res.headers.entries()));
    
    if (!res.ok) {
      const error = await res.text();
      console.error(`[API] Error response:`, error);
      throw new Error(`HTTP ${res.status}: ${error}`);
    }
    
    console.log(`[API] Parsing JSON response...`);
    const data = await res.json();
    console.log(`[API] Response received:`, {
      run_id: data.run_id,
      n_masks: data.mask_assignments?.length,
      image_size: data.image?.length,
    });
    
    return data;
  } catch (e) {
    clearTimeout(timeoutId);
    
    console.error(`[API] Fetch error:`, e);
    
    // Handle abort (timeout)
    if (e.name === 'AbortError') {
      throw new Error(
        `Request timed out after 3 minutes. The backend is processing a large video. ` +
        `Please wait and try again, or check the backend logs.`
      );
    }
    
    // Improve error message for network/CORS errors
    if (e instanceof TypeError && e.message.includes("fetch")) {
      // Check if it's a CORS error specifically
      if (e.message.includes("CORS") || e.message.includes("cors")) {
        throw new Error(
          `CORS error: The backend is not allowing requests from this origin. ` +
          `Make sure CORS middleware is enabled on the backend. ` +
          `Backend: ${BACKEND}`
        );
      }
      throw new Error(
        `Network error: Could not connect to backend at ${BACKEND}. ` +
        `Make sure: 1) The server is running, 2) SSH port forwarding is active ` +
        `(ssh -N -L 12212:r02g03.bullx:12212 gregormi@puhti.csc.fi), ` +
        `3) CORS is enabled. Original error: ${e.message}`
      );
    }
    throw e;
  }
}

/**
 * Prepare a new annotation session WITHOUT running SAM.
 * This extracts frames and returns run_id + metadata + a source preview URL.
 * @param {string} videoPath
 * @returns {Promise<{run_id: string, fps: number, n_frames_total: number, width: number|null, height: number|null, source_url: string}>}
 */
export async function prepareVideo(videoPath) {
  const params = new URLSearchParams({ video_path: videoPath });

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 min
  try {
    const res = await fetch(`${BACKEND}/prepare?${params}`, {
      method: "POST",
      signal: controller.signal,
      headers: { "Accept": "application/json" },
    });
    clearTimeout(timeoutId);
    if (!res.ok) {
      const error = await res.text();
      throw new Error(`HTTP ${res.status}: ${error}`);
    }
    return await res.json();
  } catch (e) {
    clearTimeout(timeoutId);
    if (e.name === "AbortError") {
      throw new Error(
        `Request timed out after 3 minutes. The backend is preparing/extracting frames. ` +
          `Please wait and try again, or check backend logs.`
      );
    }
    throw e;
  }
}

/**
 * Get progress for prepare_upload operation.
 * @param {string} runId
 * @returns {Promise<{status: string, stage?: string, progress: number, message: string}>}
 */
export async function getPrepareUploadProgress(runId) {
  try {
    const res = await fetch(`${BACKEND}/prepare_upload_progress/${runId}`);
    if (!res.ok) {
      console.warn(`[API] Progress request failed: ${res.status}`);
      return { status: "error", progress: 0, message: "Failed to get progress" };
    }
    const data = await res.json();
    console.log(`[API] Progress response:`, data);
    return data;
  } catch (e) {
    console.error(`[API] Progress request error:`, e);
    return { status: "error", progress: 0, message: e.message };
  }
}

/**
 * Upload a local video file to backend and prepare a run (extract frames).
 * @param {File} file
 * @param {Function} onProgress - Optional callback(progress, message, stage) for progress updates
 * @returns {Promise<{run_id: string, fps: number, n_frames_total: number, width: number|null, height: number|null, source_url: string, uploaded_filename: string}>}
 */
export async function prepareUploadVideo(file, onProgress = null) {
  console.log("[API] Starting prepareUploadVideo, file size:", file.size);
  const form = new FormData();
  form.append("file", file);

  // Poll for progress if callback provided
  let pollInterval = null;
  let runId = null;
  let pollActive = false;
  
  if (onProgress) {
    console.log("[API] Progress callback provided");
    onProgress(0, "Starting upload...", "upload");
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
    if (pollInterval) clearInterval(pollInterval);
  }, 180000); // 3 min
  
  try {
    console.log("[API] Starting upload request...");
    const res = await fetch(`${BACKEND}/prepare_upload`, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    
    console.log("[API] Upload response received, status:", res.status);
    
    if (!res.ok) {
      const error = await res.text();
      console.error("[API] Upload failed:", error);
      if (pollInterval) clearInterval(pollInterval);
      throw new Error(`HTTP ${res.status}: ${error}`);
    }
    
    // Read response once
    const result = await res.json();
    console.log("[API] Upload response parsed, run_id:", result.run_id, "status:", result.status);
    runId = result.run_id;
    
    // If extraction is in progress, we need to poll for final results
    if (result.status === "uploaded") {
      console.log("[API] Extraction in progress, will poll for completion");
    }
    
    // Start polling for progress if callback provided
    // Note: The upload might be done, but extraction might still be in progress
    if (onProgress && runId) {
      console.log("[API] Starting progress polling for run_id:", runId);
      pollActive = true;
      
      // Poll immediately first time
      const pollOnce = async () => {
        if (!pollActive || !runId) return;
        try {
          const progress = await getPrepareUploadProgress(runId);
          console.log("[API] Progress update:", progress);
          if (onProgress) {
            onProgress(progress.progress, progress.message, progress.stage);
          }
          
          if (progress.status === "completed") {
            console.log("[API] Progress completed, stopping polling");
            pollActive = false;
            if (pollInterval) clearInterval(pollInterval);
            if (onProgress) {
              onProgress(100, "Completed", null);
            }
            // Update result with final metadata if available
            if (progress.fps !== undefined) result.fps = progress.fps;
            if (progress.n_frames_total !== undefined) result.n_frames_total = progress.n_frames_total;
            if (progress.width !== undefined) result.width = progress.width;
            if (progress.height !== undefined) result.height = progress.height;
          }
        } catch (e) {
          console.error("[API] Progress polling error:", e);
          // Continue polling even on error
        }
      };
      
      // Poll immediately
      pollOnce();
      
      // Then poll every 200ms
      pollInterval = setInterval(pollOnce, 200);
      
      // Clear interval after 3 minutes max
      setTimeout(() => {
        if (pollInterval) {
          console.log("[API] Progress polling timeout, clearing interval");
          pollActive = false;
          clearInterval(pollInterval);
        }
      }, 180000);
    } else if (onProgress) {
      onProgress(100, "Completed", null);
    }
    
    // If extraction is complete, wait a bit for final metadata
    if (result.status === "uploaded") {
      // Wait for extraction to complete (poll until we get final metadata)
      let attempts = 0;
      while (attempts < 300 && (!result.fps || !result.n_frames_total)) {  // Max 60 seconds
        await new Promise(resolve => setTimeout(resolve, 200));
        try {
          const progress = await getPrepareUploadProgress(runId);
          if (progress.status === "completed" && progress.fps !== undefined) {
            result.fps = progress.fps;
            result.n_frames_total = progress.n_frames_total;
            result.width = progress.width;
            result.height = progress.height;
            break;
          }
        } catch (e) {
          // Continue waiting
        }
        attempts++;
      }
    }
    
    return result;
  } catch (e) {
    clearTimeout(timeoutId);
    if (pollInterval) {
      pollActive = false;
      clearInterval(pollInterval);
    }
    console.error("[API] Upload error:", e);
    if (e.name === "AbortError") {
      throw new Error(
        `Request timed out after 3 minutes. The backend is uploading/preparing frames. ` +
          `Please wait and try again, or check backend logs.`
      );
    }
    throw e;
  }
}

/**
 * Run SAM on frame 0 for an existing prepared run_id.
 * @param {string} runId
 * @param {string} prompt
 */
export async function initSam(runId, prompt = "cow") {
  const params = new URLSearchParams({ prompt });
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 min
  try {
    const res = await fetch(`${BACKEND}/init_sam/${runId}?${params}`, {
      method: "POST",
      signal: controller.signal,
      headers: { "Accept": "application/json" },
    });
    clearTimeout(timeoutId);
    if (!res.ok) {
      const error = await res.text();
      throw new Error(`HTTP ${res.status}: ${error}`);
    }
    return await res.json();
  } catch (e) {
    clearTimeout(timeoutId);
    if (e.name === "AbortError") {
      throw new Error(
        `Request timed out after 3 minutes. The backend is running SAM. ` +
          `Please wait and try again, or check backend logs.`
      );
    }
    throw e;
  }
}

export function getSourceVideoUrl(runId) {
  return `${BACKEND}/source/${runId}`;
}

/**
 * Apply ID mappings to complete initialization
 * @param {string} runId - Run ID from init
 * @param {Object} mapping - Object mapping mask_index (string) to final_id (number)
 * @returns {Promise<{run_id: string, n_ids: number}>}
 */
export async function applyInitIds(runId, mapping, behaviorByCowId = null) {
  const body = { mapping };
  if (behaviorByCowId && Object.keys(behaviorByCowId).length > 0) {
    body.behavior_by_cow_id = behaviorByCowId;
  }
  const res = await fetch(`${BACKEND}/apply_init_ids/${runId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error);
  }
  return res.json();
}

export async function setAnnotationMode(runId, mode) {
  const res = await fetch(`${BACKEND}/run/${runId}/annotation_mode`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`HTTP ${res.status}: ${error}`);
  }
  return res.json();
}

export async function getBehavior(runId, frame = null) {
  const params = frame !== null ? `?frame=${frame}` : "";
  const res = await fetch(`${BACKEND}/behavior/${runId}${params}`);
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`HTTP ${res.status}: ${error}`);
  }
  return res.json();
}

export async function setBehaviorLabel(runId, cowId, frame, labelId) {
  const res = await fetch(`${BACKEND}/behavior/${runId}/set_label`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cow_id: cowId, frame, label_id: labelId }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`HTTP ${res.status}: ${error}`);
  }
  return res.json();
}

export async function rebuildGoldenPreview(runId) {
  const res = await fetch(`${BACKEND}/golden/${runId}/rebuild_preview`, {
    method: "POST",
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`HTTP ${res.status}: ${error}`);
  }
  return res.json();
}

/**
 * Track frames
 * @param {string} runId - Run ID
 * @param {number} nFrames - Number of frames to track
 * @param {number|null} autoResetInterval - Optional auto-reset interval
 * @returns {Promise<{run_id: string, ...}>}
 */
export async function trackFrames(runId, nFrames, autoResetInterval = null, onProgress = null) {
  const params = new URLSearchParams({
    run_id: runId,
    n_frames: String(nFrames),
  });
  if (autoResetInterval !== null && autoResetInterval > 0) {
    params.append("auto_reset_interval", String(autoResetInterval));
  }

  // Poll for progress if callback provided
  let pollInterval = null;
  let pollActive = false;
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 3600000); // 1 hour timeout for tracking

  try {
    // Start the fetch request first
    const fetchPromise = fetch(`${BACKEND}/track?${params}`, {
      method: "POST",
      signal: controller.signal,
    });
    
    // Start polling immediately to catch initial progress
    if (onProgress) {
      console.log("[API] Starting tracking progress polling for run_id:", runId);
      pollActive = true;
      
      const pollOnce = async () => {
        if (!pollActive || !runId) return;
        try {
          const progress = await getTrackProgress(runId);
          console.log("[API] Track progress update:", progress);
          
          // Update UI if tracking has started or is in progress
          if (progress.status === "in_progress" || progress.status === "completed") {
            if (onProgress) {
              onProgress(progress.progress, progress.message, progress.stage);
            }
          } else if (progress.status === "not_started") {
            // Show 0% progress while waiting
            if (onProgress) {
              onProgress(0, "Waiting to start...", "tracking");
            }
          }
          
          if (progress.status === "completed") {
            console.log("[API] Tracking completed, stopping polling");
            pollActive = false;
            if (pollInterval) clearInterval(pollInterval);
            if (onProgress) {
              onProgress(100, "Completed", null);
            }
          }
        } catch (e) {
          console.error("[API] Track progress polling error:", e);
          // Continue polling even on error
        }
      };
      
      // Poll immediately (backend initializes progress at the start of /track)
      pollOnce();
      
      // Then poll every 500ms (tracking is slower than frame extraction)
      pollInterval = setInterval(pollOnce, 500);
      
      // Clear interval after 1 hour max
      setTimeout(() => {
        if (pollInterval) {
          console.log("[API] Track progress polling timeout, clearing interval");
          pollActive = false;
          clearInterval(pollInterval);
        }
      }, 3600000);
    }
    
    const res = await fetchPromise;
    clearTimeout(timeoutId);

    if (!res.ok) {
      const error = await res.text();
      if (pollInterval) {
        pollActive = false;
        clearInterval(pollInterval);
      }
      throw new Error(`HTTP ${res.status}: ${error}`);
    }
    
    const result = await res.json();
    
    // Stop polling after response
    if (pollInterval) {
      pollActive = false;
      clearInterval(pollInterval);
    }
    
    return result;
  } catch (e) {
    clearTimeout(timeoutId);
    if (pollInterval) {
      pollActive = false;
      clearInterval(pollInterval);
    }
    if (e.name === "AbortError") {
      throw new Error("Tracking timed out after 1 hour. Check backend logs.");
    }
    throw e;
  }
}

/**
 * Commit tracked frames to golden
 * @param {string} runId - Run ID
 * @returns {Promise<{run_id: string, ...}>}
 */
export async function commitFrames(runId) {
  const params = new URLSearchParams({ run_id: runId });
  const res = await fetch(`${BACKEND}/commit?${params}`, {
    method: "POST",
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`HTTP ${res.status}: ${error}`);
  }
  return res.json();
}

/**
 * Get tracking progress
 * @param {string} runId - Run ID
 * @returns {Promise<{status: string, stage: string, progress: number, message: string, ...}>}
 */
export async function getTrackProgress(runId) {
  try {
    const res = await fetch(`${BACKEND}/track_progress/${runId}`);
    if (!res.ok) {
      return { status: "error", progress: 0, message: "Failed to get progress" };
    }
    return await res.json();
  } catch (e) {
    return { status: "error", progress: 0, message: e.message };
  }
}

/**
 * Get progress information
 * @param {string} runId - Run ID
 * @returns {Promise<{golden_processed: number, golden_percent: number, ...}>}
 */
export async function getProgress(runId) {
  const res = await fetch(`${BACKEND}/progress/${runId}`);
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`HTTP ${res.status}: ${error}`);
  }
  return res.json();
}

/**
 * Get video URL for tracked result
 */
export function getTrackedVideoUrl(runId) {
  return `${BACKEND}/result/${runId}`;
}

/**
 * Get video URL for golden preview
 */
export function getGoldenVideoUrl(runId) {
  return `${BACKEND}/golden_video/${runId}`;
}

/**
 * Download the golden folder as a zip file
 * @param {string} runId - The run ID
 */
export async function downloadGolden(runId) {
  const url = `${BACKEND}/download_golden/${runId}`;
  
  try {
    // Important: the golden zip can be very large (hundreds of MB).
    // Fetching it into memory via `res.blob()` is prone to failing with
    // a generic "Failed to fetch" / OOM. Let the browser handle the download.
    const link = document.createElement("a");
    link.href = url;
    link.rel = "noopener";
    // `download` may be ignored for cross-origin URLs, but backend sets Content-Disposition anyway.
    link.download = `${runId}_golden.zip`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  } catch (e) {
    console.error("[API] Failed to download golden folder:", e);
    throw e;
  }
}

/**
 * Get a frame from the tracked video
 * @param {string} runId - Run ID
 * @param {number} relativeFrameIdx - Relative frame index in the tracked chunk
 * @returns {Promise<Blob>} - Image blob
 */
export async function getTrackedFrame(runId, relativeFrameIdx) {
  const res = await fetch(`${BACKEND}/tracked_frame/${runId}/${relativeFrameIdx}`);
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`HTTP ${res.status}: ${error}`);
  }
  return res.blob();
}

/**
 * Prepare correction for a frame
 * @param {string} runId - Run ID
 * @param {number} frameIdx - Absolute frame index to correct
 * @returns {Promise<{image: string, mask_assignments: Array, existing_ids: Array}>}
 */
export async function prepareCorrection(runId, frameIdx) {
  const res = await fetch(`${BACKEND}/prepare_correction/${runId}/${frameIdx}`, {
    method: "POST",
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`HTTP ${res.status}: ${error}`);
  }
  return res.json();
}

/**
 * Preview correction update (update preview when IDs change)
 * @param {string} runId - Run ID
 * @param {number} frameIdx - Absolute frame index
 * @param {Object} mapping - ID mapping
 * @returns {Promise<{image: string}>}
 */
export async function previewCorrectionUpdate(runId, frameIdx, mapping) {
  const res = await fetch(`${BACKEND}/preview_correction_update/${runId}/${frameIdx}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mapping }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`HTTP ${res.status}: ${error}`);
  }
  return res.json();
}

/**
 * Apply correction
 * @param {string} runId - Run ID
 * @param {number} frameIdx - Absolute frame index
 * @param {Object} mapping - ID mapping
 * @returns {Promise<{status: string, max_id: number}>}
 */
export async function applyCorrection(runId, frameIdx, mapping) {
  const res = await fetch(`${BACKEND}/apply_correction/${runId}/${frameIdx}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mapping }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`HTTP ${res.status}: ${error}`);
  }
  return res.json();
}

/**
 * Refine a mask using point prompts
 * @param {string} runId - Run ID
 * @param {number} frameIdx - Absolute frame index
 * @param {number} maskIndex - Index of mask to refine
 * @param {Array} points - Array of {x, y, is_positive} points
 * @returns {Promise<{image: string, mask_index: number, refined_mask_size: number, image_width: number, image_height: number}>}
 */
export async function refineMask(runId, frameIdx, maskIndex, points) {
  const res = await fetch(`${BACKEND}/refine_mask/${runId}/${frameIdx}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mask_index: maskIndex,
      points: points.map((p) => ({
        x: p.x,
        y: p.y,
        is_positive: p.is_positive,
      })),
    }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`HTTP ${res.status}: ${error}`);
  }
  return res.json();
}

/**
 * Add a new mask using a point prompt
 * @param {string} runId - Run ID
 * @param {number} frameIdx - Absolute frame index
 * @param {number} x - X coordinate
 * @param {number} y - Y coordinate
 * @param {boolean} isPositive - Whether point is positive (should be true for new masks)
 * @returns {Promise<{image: string, new_mask_index: number, new_mask_size: number, total_masks: number, image_width: number, image_height: number}>}
 */
export async function addMask(runId, frameIdx, x, y, isPositive) {
  const res = await fetch(`${BACKEND}/add_mask/${runId}/${frameIdx}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ point: { x, y, is_positive: isPositive } }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`HTTP ${res.status}: ${error}`);
  }
  return res.json();
}

/**
 * Match IDs from a previous mask file
 * @param {string} runId - Run ID
 * @param {File} maskFile - The uploaded PNG mask file
 * @returns {Promise<{mask_assignments: Array, matched_count: number, total_count: number, image: string}>}
 */
export async function matchInitIds(runId, maskFile) {
  const formData = new FormData();
  formData.append('file', maskFile);
  
  const res = await fetch(`${BACKEND}/match_init_ids/${runId}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`HTTP ${res.status}: ${error}`);
  }
  return res.json();
}

/**
 * Preview init update (update preview when IDs change in initial assignment)
 * @param {string} runId - Run ID
 * @param {Object} mapping - ID mapping
 * @returns {Promise<{image: string}>}
 */
export async function previewInitUpdate(runId, mapping) {
  const res = await fetch(`${BACKEND}/preview_init_update/${runId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mapping }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`HTTP ${res.status}: ${error}`);
  }
  return res.json();
}
