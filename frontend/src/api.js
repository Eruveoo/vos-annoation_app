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
 * Apply ID mappings to complete initialization
 * @param {string} runId - Run ID from init
 * @param {Object} mapping - Object mapping mask_index (string) to final_id (number)
 * @returns {Promise<{run_id: string, n_ids: number}>}
 */
export async function applyInitIds(runId, mapping) {
  const res = await fetch(`${BACKEND}/apply_init_ids/${runId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mapping }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error);
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
export async function trackFrames(runId, nFrames, autoResetInterval = null) {
  const params = new URLSearchParams({
    run_id: runId,
    n_frames: String(nFrames),
  });
  if (autoResetInterval !== null && autoResetInterval > 0) {
    params.append("auto_reset_interval", String(autoResetInterval));
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 3600000); // 1 hour timeout for tracking

  try {
    const res = await fetch(`${BACKEND}/track?${params}`, {
      method: "POST",
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      const error = await res.text();
      throw new Error(`HTTP ${res.status}: ${error}`);
    }
    return res.json();
  } catch (e) {
    clearTimeout(timeoutId);
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
    const res = await fetch(url);
    if (!res.ok) {
      const error = await res.text();
      throw new Error(`HTTP ${res.status}: ${error}`);
    }
    
    // Get the blob and create a download link
    const blob = await res.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `${runId}_golden.zip`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(downloadUrl);
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
