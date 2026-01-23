import tempfile
import gradio as gr
import requests
import os
import time

API = "http://127.0.0.1:12212"

CSS = """
#source_video video  { height: 854px !important; width: 100% !important; object-fit: contain; }
#tracked_video video { height: 854px !important; width: 100% !important; object-fit: contain; }
#golden_video video  { height: 854px !important; width: 100% !important; object-fit: contain; }
"""


def _download_to_temp(url: str, suffix: str):
    url = f"{url}?t={time.time_ns()}"   # cache-buster
    r = requests.get(url, timeout=3600)
    r.raise_for_status()
    _, path = tempfile.mkstemp(suffix=suffix)
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def _get_progress(run_id: str):
    r = requests.get(f"{API}/progress/{run_id}", timeout=60)
    r.raise_for_status()
    return r.json()


def _get_paths(run_id: str):
    r = requests.get(f"{API}/paths/{run_id}", timeout=60)
    r.raise_for_status()
    return r.json()


def _progress_text(p: dict) -> str:
    total = p.get("total_frames", 0)
    processed = p.get("golden_processed", 0)
    pct = float(p.get("golden_percent", 0.0))
    return f"**Golden progress:** {processed}/{total} frames ({pct:.1f}%)"


def init_and_load(video_path, prompt):
    """Initialize video and return SAM masks for ID assignment."""
    r = requests.post(
        f"{API}/init",
        params={"video_path": video_path, "prompt": prompt},
        timeout=3600,
    )
    r.raise_for_status()
    data = r.json()
    run_id = data["run_id"]

    # Decode preview image with SAM masks
    import numpy as np, cv2
    import base64
    image_data = data["image"].split(",")[1]  # Remove data:image/jpeg;base64, prefix
    img_bytes = base64.b64decode(image_data)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Build dataframe for ID mapping
    mask_assignments = data["mask_assignments"]
    df_data = [
        [m["mask_index"], m["auto_assigned_id"], m["auto_assigned_id"], False]  # Mask #, Auto ID, Final ID, Delete?
        for m in mask_assignments
    ]

    return (
        run_id,
        im,  # frame0 with SAM masks
        df_data,  # id_mapping_table for init
        gr.update(visible=True),  # init_id_section (shows the Row with both frame0 and table)
        gr.update(visible=True),  # frame0 (show the preview within the Row)
    )


def apply_init_ids_wrapper(run_id_val, id_mapping_df):
    """Apply user's ID mapping to complete initialization."""
    if not run_id_val:
        raise gr.Error("No run_id yet.")
    
    if id_mapping_df is None or len(id_mapping_df) == 0:
        raise gr.Error("No ID mappings provided.")
    
    # Convert dataframe to dict: mask_index -> final_id
    import pandas as pd
    id_mapping = {}
    if isinstance(id_mapping_df, pd.DataFrame):
        for _, row in id_mapping_df.iterrows():
            mask_idx = int(row["Mask #"])
            final_id = int(row["Final ID"])
            delete = bool(row.get("Delete?", False))
            if not delete and final_id > 0:
                id_mapping[str(mask_idx)] = final_id
    else:
        raise gr.Error(f"Unexpected dataframe type: {type(id_mapping_df)}")
    
    # Call apply_init_ids endpoint
    import json
    r = requests.post(
        f"{API}/apply_init_ids/{run_id_val}",
        json={"mapping": id_mapping},
        timeout=3600,
    )
    r.raise_for_status()
    data = r.json()
    
    # Now load the completed initialization
    source_path = _download_to_temp(f"{API}/source/{run_id_val}", suffix=".mp4")
    golden_path = _download_to_temp(f"{API}/golden_video/{run_id_val}", suffix=".mp4")
    
    # Get frame0 image (now with final IDs)
    img = requests.get(f"{API}/frame0/{run_id_val}", timeout=300)
    img.raise_for_status()
    
    import numpy as np, cv2
    arr = np.frombuffer(img.content, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    prog = _get_progress(run_id_val)
    paths = _get_paths(run_id_val)
    
    return (
        source_path,
        im,  # frame0 with final IDs
        golden_path,
        paths.get("golden_annotations", ""),
        _progress_text(prog),
        float(prog.get("golden_percent", 0.0)),
        gr.update(visible=False),  # Hide init_id_section
    )


def resume_and_load(resume_run_id):
    """Resume an existing annotation session."""
    if not resume_run_id or resume_run_id.strip() == "":
        raise gr.Error("Please enter a run_id to resume (e.g., 20260120_171846_4dd2ab)")
    
    resume_run_id = resume_run_id.strip()
    
    # Resume the session
    r = requests.post(
        f"{API}/resume",
        params={"run_id": resume_run_id},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    run_id = data["run_id"]

    # frame0
    img = requests.get(f"{API}/frame0/{run_id}", timeout=300)
    img.raise_for_status()

    import numpy as np, cv2
    arr = np.frombuffer(img.content, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    source_path = _download_to_temp(f"{API}/source/{run_id}", suffix=".mp4")
    golden_path = _download_to_temp(f"{API}/golden_video/{run_id}", suffix=".mp4")

    prog = _get_progress(run_id)
    paths = _get_paths(run_id)

    return (
        run_id,
        source_path,
        im,
        golden_path,
        paths.get("golden_annotations", ""),
        _progress_text(prog),
        float(prog.get("golden_percent", 0.0)),
    )


def do_track(run_id, n_frames, auto_reset_interval):
    if not run_id:
        raise gr.Error("No run_id yet.")

    params = {"run_id": run_id, "n_frames": int(n_frames)}
    if auto_reset_interval is not None and auto_reset_interval != "":
        try:
            reset_val = int(auto_reset_interval)
            if reset_val > 0:
                params["auto_reset_interval"] = reset_val
                print(f"[DEBUG UI] Auto-reset interval set to: {reset_val}")
        except (ValueError, TypeError) as e:
            print(f"[DEBUG UI] Invalid auto_reset_interval value: {auto_reset_interval}, error: {e}")
            pass  # Ignore invalid values, just don't use auto-reset
    else:
        print(f"[DEBUG UI] Auto-reset interval not set (value: {auto_reset_interval})")
    
    print(f"[DEBUG UI] Track params: {params}")
    r = requests.post(
        f"{API}/track",
        params=params,
        timeout=3600,
    )
    r.raise_for_status()

    tracked_path = _download_to_temp(f"{API}/result/{run_id}", suffix=".mp4")
    prog = _get_progress(run_id)

    return tracked_path, _progress_text(prog), float(prog.get("golden_percent", 0.0))


def do_commit(run_id):
    if not run_id:
        raise gr.Error("No run_id yet.")

    r = requests.post(f"{API}/commit", params={"run_id": run_id}, timeout=3600)
    r.raise_for_status()

    golden_path = _download_to_temp(f"{API}/golden_video/{run_id}", suffix=".mp4")
    prog = _get_progress(run_id)

    return golden_path, f"✅ Committed. {_progress_text(prog)}", float(prog.get("golden_percent", 0.0))


def do_save(run_id):
    """Create a backup/snapshot of the current run."""
    if not run_id:
        raise gr.Error("No run_id yet. Load or resume a session first.")

    r = requests.post(f"{API}/save", params={"run_id": run_id}, timeout=300)
    r.raise_for_status()
    data = r.json()
    backup_run_id = data["backup_run_id"]
    original_run_id = data["original_run_id"]

    return f"💾 Backup created!\n\n**Backup run_id:** `{backup_run_id}`\n**Original run_id:** `{original_run_id}`\n\nYou can resume from this backup later if needed."


def do_correct_frame(run_id, wrong_frame_idx, fps=None):
    """Correct frame with optional auto-detection."""
    if not run_id:
        raise gr.Error("No run_id yet.")
    
    # If frame_idx is None, try to get it from video (will be handled by JS)
    if wrong_frame_idx is None or wrong_frame_idx == "":
        raise gr.Error("Please specify the frame index to correct, or use 'Correct current frame' button.")
    
    try:
        frame_idx = int(wrong_frame_idx)
    except ValueError:
        raise gr.Error(f"Invalid frame index: {wrong_frame_idx}")
    
    # Call correct_frame endpoint
    r = requests.post(
        f"{API}/correct_frame",
        params={"run_id": run_id, "wrong_frame_idx": frame_idx},
        timeout=3600,
    )
    r.raise_for_status()
    
    # Get the corrected frame image
    import numpy as np, cv2
    arr = np.frombuffer(r.content, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # Update progress
    prog = _get_progress(run_id)
    golden_path = _download_to_temp(f"{API}/golden_video/{run_id}", suffix=".mp4")
    
    return (
        im,
        golden_path,
        f"✅ Corrected frame {frame_idx}. Committed frames up to {frame_idx-1}. {_progress_text(prog)}",
        float(prog.get("golden_percent", 0.0)),
    )




with gr.Blocks() as demo:
    gr.Markdown("## VOS v1 — load → track → commit → progress")

    with gr.Row():
        video_path = gr.Textbox(label="Video path on Puhti", value="video_sample_5_min.mp4", scale=3)
        prompt = gr.Textbox(label="Text prompt", value="cow", scale=1)

    with gr.Row():
        load_btn = gr.Button("🆕 Load video + run prompt on frame 0", variant="primary")
        resume_run_id = gr.Textbox(
            label="Resume existing session (run_id)",
            placeholder="e.g., 20260120_171846_4dd2ab",
            scale=2,
            info="Enter the run_id from a previous session to continue where you left off",
        )
        resume_btn = gr.Button("▶️ Resume session", variant="secondary")
    
    run_id = gr.Textbox(label="run_id", interactive=False)

    golden_path_box = gr.Textbox(
        label="Golden annotations folder (download from Puhti)",
        interactive=False,
    )

    # Original video
    with gr.Row():
        source_video = gr.Video(
            label="Original video (preview)",
            height=854,
            autoplay=False,
            elem_id="source_video",
        )
    
    # Frame 0 SAM preview and ID assignment table side-by-side
    with gr.Row(visible=False) as init_id_section:
        with gr.Column():
            frame0 = gr.Image(label="Frame 0 (SAM-3 result)", type="numpy", visible=False)
        with gr.Column():
            gr.Markdown("### Assign IDs to detected masks")
            gr.Markdown("SAM detected masks with auto-assigned IDs. Review and change IDs if needed (e.g., to continue from a previous video segment).")
            
            # Option to match IDs from previous golden annotation
            with gr.Row():
                previous_mask_file = gr.File(
                    label="Upload previous golden mask file (optional)",
                    file_types=[".png"],
                    scale=3,
                )
                match_ids_btn = gr.Button("🔗 Match IDs from previous masks", variant="secondary", scale=1)
            gr.Markdown("*Upload the last frame mask PNG file from a previous annotation session (e.g., 00123.png from golden/Annotations/video1/)*", elem_classes=["small-text"])
            
            init_id_mapping_table = gr.Dataframe(
                headers=["Mask #", "Auto-assigned ID", "Final ID", "Delete?"],
                datatype=["number", "number", "number", "bool"],
                label="ID assignments",
                interactive=True,
                wrap=True,
            )
            
            apply_init_ids_btn = gr.Button("✅ Apply IDs and complete initialization", variant="primary")
            match_status = gr.Markdown("", visible=False)

    n_frames = gr.Slider(1, 5000, value=50, step=1, label="N new frames to track")
    auto_reset_interval = gr.Number(
        label="Auto-reset interval (optional)",
        value=10,
        precision=0,
        info="Automatically reinitializes with SAM every K frames to handle drift (e.g., 10 = reset on frames 10, 20, 30, etc.). Leave empty to disable.",
    )

    with gr.Row():
        track_btn = gr.Button("Track + render chunk")
        commit_btn = gr.Button("✅ Commit chunk to golden")
        save_btn = gr.Button("💾 Save backup", variant="secondary")

    progress_md = gr.Markdown("**Golden progress:** —")
    progress_bar = gr.Slider(0, 100, value=0, step=0.1, interactive=False, label="Progress (%)")
    save_status = gr.Markdown("", visible=False)
    
    # Correction section
    gr.Markdown("---")
    gr.Markdown("### Correction: Select and correct wrong frame")
    gr.Markdown("**Step 1:** Preview frames to find the wrong one. **Step 2:** Prepare correction to run SAM. **Step 3:** Review and reassign IDs.")
    
    with gr.Row():
        with gr.Column(scale=1):
            preview_frame_idx = gr.Number(
                label="Frame number to preview",
                value=None,
                precision=0,
                info="Relative frame number (0 = first frame of chunk). Preview to find the wrong frame.",
            )
            get_frame_btn = gr.Button("👁️ Preview frame", variant="secondary")
        with gr.Column(scale=1):
            wrong_frame_idx = gr.Number(
                label="Frame index to correct",
                value=None,
                precision=0,
                info="Frame number relative to the start of current tracked chunk. Auto-filled when you preview.",
            )
            prepare_correction_btn = gr.Button("🔧 Prepare correction (commit before + run SAM)", variant="primary")
    
    correction_status = gr.Markdown("")

    # Tracked video with previewed frame side-by-side
    with gr.Row():
        with gr.Column():
            out_video = gr.Video(
                label="Tracked chunk preview (current)",
                height=854,
                autoplay=False,
                elem_id="tracked_video",
            )
        with gr.Column():
            previewed_frame = gr.Image(
                label="Previewed frame (from tracked chunk)",
                type="numpy",
                visible=False,
            )
            corrected_frame = gr.Image(
                label="Corrected frame (after applying your ID mapping)",
                type="numpy",
                visible=False,
            )
            corrected_frame_info = gr.Markdown("")
    
    # SAM preview frame and ID reassignment table side-by-side (below tracked video)
    with gr.Row():
        with gr.Column():
            correction_preview_frame = gr.Image(
                label="Frame with SAM masks and auto-assigned IDs",
                type="numpy",
                visible=False,
                interactive=True,  # Enable interaction for click events
            )
        with gr.Column(visible=False) as id_reassignment_section:
            gr.Markdown("### Review and reassign IDs")
            gr.Markdown("SAM detected masks with auto-assigned IDs. Review and change IDs if needed (e.g., if a cow reappeared).")
            
            # Mask refinement controls
            gr.Markdown("### Refine masks with point prompts")
            gr.Markdown("**Instructions:** 1) Select mask number, 2) Choose Add/Remove mode, 3) Click directly on the image to refine the mask")
            with gr.Row():
                selected_mask_idx = gr.Number(
                    label="Mask to refine",
                    value=None,
                    precision=0,
                    info="Mask number (0, 1, 2, ...) - click on image after selecting",
                )
                point_mode = gr.Radio(
                    choices=["Add points", "Remove points"],
                    value="Add points",
                    label="Point mode",
                    info="Add: click to include in mask. Remove: click to exclude from mask.",
                )
            refinement_status = gr.Markdown("", visible=False)
            
            id_mapping_table = gr.Dataframe(
                headers=["Mask #", "Auto-assigned ID", "Final ID", "Delete?"],
                datatype=["number", "number", "number", "bool"],
                label="ID assignments",
                interactive=True,
                wrap=True,
            )
            
            with gr.Row():
                apply_correction_btn = gr.Button("✅ Apply correction", variant="primary")
                cancel_correction_btn = gr.Button("❌ Cancel", variant="secondary")

    # Golden video below
    golden_video = gr.Video(
        label="Golden preview (committed so far)",
        height=854,
        autoplay=False,
        elem_id="golden_video",
    )

    load_btn.click(
        init_and_load,
        [video_path, prompt],
        [run_id, frame0, init_id_mapping_table, init_id_section, frame0],
    )
    
    # Match IDs from previous masks
    def match_ids_from_previous(run_id_val, previous_mask_file_val):
        """Match new SAM masks to IDs from a previous golden mask file."""
        if not run_id_val:
            raise gr.Error("No run_id yet. Please load a video first.")
        
        if previous_mask_file_val is None:
            raise gr.Error("Please upload a previous mask file.")
        
        try:
            # Upload the file to backend
            # Gradio File component returns a file path string
            mask_file_path = previous_mask_file_val if isinstance(previous_mask_file_val, str) else previous_mask_file_val.name if hasattr(previous_mask_file_val, 'name') else None
            
            if mask_file_path is None or not os.path.exists(mask_file_path):
                raise gr.Error(f"Mask file not found: {mask_file_path}")
            
            with open(mask_file_path, 'rb') as f:
                files = {'file': ('mask.png', f, 'image/png')}
                r = requests.post(
                    f"{API}/match_init_ids/{run_id_val}",
                    files=files,
                    timeout=60,
                )
            r.raise_for_status()
            data = r.json()
            
            # Build dataframe with matched IDs
            mask_assignments = data["mask_assignments"]
            df_data = [
                [m["mask_index"], m["auto_assigned_id"], m["matched_id"], False]  # Use matched_id as Final ID
                for m in mask_assignments
            ]
            
            # Update preview image if provided
            preview_im = None
            if "image" in data:
                import numpy as np, cv2
                import base64
                image_data = data["image"].split(",")[1]
                img_bytes = base64.b64decode(image_data)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                preview_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            matched_count = data.get("matched_count", 0)
            total_count = data.get("total_count", 0)
            status_msg = f"✅ Matched {matched_count}/{total_count} masks to previous IDs!"
            
            if preview_im is not None:
                return (
                    preview_im,  # Updated frame0 with matched IDs
                    df_data,  # Updated table
                    gr.update(visible=True, value=status_msg),  # Status
                )
            else:
                return (
                    gr.update(),  # No preview update
                    df_data,  # Updated table
                    gr.update(visible=True, value=status_msg),  # Status
                )
        except Exception as e:
            error_msg = f"❌ Failed to match IDs: {str(e)}"
            print(f"[DEBUG UI] Error matching IDs: {e}")
            import traceback
            print(f"[DEBUG UI] Traceback: {traceback.format_exc()}")
            return (
                gr.update(),  # No preview update
                gr.update(),  # No table update
                gr.update(visible=True, value=error_msg),  # Error status
            )
    
    match_ids_btn.click(
        match_ids_from_previous,
        [run_id, previous_mask_file],
        [frame0, init_id_mapping_table, match_status],
    )
    
    apply_init_ids_btn.click(
        apply_init_ids_wrapper,
        [run_id, init_id_mapping_table],
        [source_video, frame0, golden_video, golden_path_box, progress_md, progress_bar, init_id_section],
    )
    
    resume_btn.click(
        resume_and_load,
        [resume_run_id],
        [run_id, source_video, frame0, golden_video, golden_path_box, progress_md, progress_bar],
    )

    track_btn.click(
        do_track,
        [run_id, n_frames, auto_reset_interval],
        [out_video, progress_md, progress_bar],
    )

    commit_btn.click(
        do_commit,
        [run_id],
        [golden_video, progress_md, progress_bar],
    )
    
    save_btn.click(
        do_save,
        [run_id],
        [save_status],
    ).then(
        lambda: gr.update(visible=True),
        outputs=[save_status],
    )
    
    # Prepare correction - runs SAM and shows auto-assigned IDs
    def prepare_correction_wrapper(run_id_val, relative_frame_idx):
        """Prepare correction: commit before, run SAM, auto-assign IDs."""
        if not run_id_val:
            raise gr.Error("No run_id yet.")
        
        if relative_frame_idx is None or relative_frame_idx == "":
            raise gr.Error("Please enter a frame index.")
        
        try:
            relative_frame = int(relative_frame_idx)
        except ValueError:
            raise gr.Error(f"Invalid frame index: {relative_frame_idx}")
        
        if relative_frame < 0:
            raise gr.Error("Frame index must be >= 0")
        
        # Get last golden frame to convert relative to absolute
        prog = _get_progress(run_id_val)
        max_idx = prog.get("golden_max_idx")
        
        if max_idx is None:
            raise gr.Error("Cannot determine last golden frame. Please track some frames first.")
        
        max_idx = int(max_idx)
        absolute_frame = max_idx + relative_frame
        
        print(f"[DEBUG] Preparing correction for frame {absolute_frame} (relative: {relative_frame})")
        
        # Call prepare_correction endpoint
        r = requests.post(
            f"{API}/prepare_correction/{run_id_val}/{absolute_frame}",
            timeout=3600,
        )
        r.raise_for_status()
        data = r.json()
        
        # Decode image
        import numpy as np, cv2
        import base64
        image_data = data["image"].split(",")[1]  # Remove data:image/jpeg;base64, prefix
        img_bytes = base64.b64decode(image_data)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        # Store actual image dimensions for coordinate scaling
        img_height, img_width = im.shape[:2]
        image_dims = (img_width, img_height)
        
        # Build dataframe for ID mapping
        mask_assignments = data["mask_assignments"]
        existing_ids = data["existing_ids"]
        
        df_data = [
            [m["mask_index"], m["auto_assigned_id"], m["auto_assigned_id"], False]  # Add Delete? column (False by default)
            for m in mask_assignments
        ]
        
        status_msg = f"✅ **Frame {absolute_frame} prepared**\n"
        status_msg += f"- Found {len(mask_assignments)} masks\n"
        status_msg += f"- Auto-assigned IDs: {[m['auto_assigned_id'] for m in mask_assignments]}\n"
        status_msg += f"- Existing IDs in sequence: {existing_ids}\n\n"
        status_msg += "**Review the IDs above and change them if needed, then click 'Apply correction'.**"
        
        # Fetch updated golden video and progress (frames were committed)
        try:
            golden_path = _download_to_temp(f"{API}/golden_video/{run_id_val}", suffix=".mp4")
            prog = _get_progress(run_id_val)
            progress_text = _progress_text(prog)
            progress_pct = float(prog.get("golden_percent", 0.0))
        except Exception as e:
            print(f"[DEBUG UI] Failed to refresh golden video: {e}")
            golden_path = None
            progress_text = status_msg
            progress_pct = None
        
        return (
            im,  # correction_preview_frame
            df_data,  # id_mapping_table
            status_msg,  # correction_status
            gr.update(visible=True),  # id_reassignment_section
            gr.update(visible=True),  # correction_preview_frame
            absolute_frame,  # Store absolute frame for apply step
            image_dims,  # Store image dimensions for coordinate scaling
            golden_path,  # golden_video (updated after commit)
            progress_text,  # progress_md
            progress_pct,  # progress_bar
        )
    
    # Update preview when table changes
    def update_preview_from_table(run_id_val, absolute_frame, id_mapping_df):
        """Update the preview image based on current table state."""
        try:
            print(f"[DEBUG UI] update_preview_from_table called: run_id={run_id_val}, absolute_frame={absolute_frame}")
            
            if not run_id_val or absolute_frame is None:
                print(f"[DEBUG UI] Skipping preview update: run_id={run_id_val}, absolute_frame={absolute_frame}")
                return gr.update()  # No update if not ready
            
            if id_mapping_df is None or len(id_mapping_df) == 0:
                print(f"[DEBUG UI] Skipping preview update: table is empty")
                return gr.update()  # No update if table is empty
            
            # Convert dataframe to mapping dict (including deletions)
            import pandas as pd
            mapping = {}
            
            if isinstance(id_mapping_df, pd.DataFrame):
                print(f"[DEBUG UI] Processing DataFrame with {len(id_mapping_df)} rows")
                print(f"[DEBUG UI] DataFrame columns: {id_mapping_df.columns.tolist()}")
                print(f"[DEBUG UI] DataFrame dtypes: {id_mapping_df.dtypes}")
                
                for idx, row in id_mapping_df.iterrows():
                    try:
                        mask_idx = int(row["Mask #"])
                        final_id = row["Final ID"]
                        
                        # Handle Delete? column - check multiple ways in case of type issues
                        delete = False
                        if "Delete?" in row:
                            delete_val = row["Delete?"]
                            # Handle different boolean representations
                            if isinstance(delete_val, bool):
                                delete = delete_val
                            elif isinstance(delete_val, (int, float)):
                                delete = bool(delete_val)
                            elif isinstance(delete_val, str):
                                delete = delete_val.lower() in ("true", "1", "yes", "on")
                            elif pd.notna(delete_val):
                                delete = bool(delete_val)
                        
                        print(f"[DEBUG UI] Row {idx}: mask_idx={mask_idx}, final_id={final_id}, delete={delete} (type: {type(delete)})")
                        
                        # If delete is True or final_id is invalid, set to 0 (delete)
                        if delete:
                            mapping[str(mask_idx)] = 0
                            print(f"[DEBUG UI] Marking mask {mask_idx} for deletion (checkbox checked)")
                        elif pd.isna(final_id) or final_id == "":
                            mapping[str(mask_idx)] = 0
                            print(f"[DEBUG UI] Marking mask {mask_idx} for deletion (empty final_id)")
                        else:
                            final_id = int(final_id)
                            if final_id < 1:
                                mapping[str(mask_idx)] = 0  # Delete if invalid
                                print(f"[DEBUG UI] Marking mask {mask_idx} for deletion (invalid ID)")
                            else:
                                mapping[str(mask_idx)] = final_id
                                print(f"[DEBUG UI] Mapping mask {mask_idx} to ID {final_id}")
                    except (ValueError, TypeError, KeyError) as e:
                        print(f"[DEBUG UI] Error parsing row {idx} for preview: {e}, row: {row}")
                        import traceback
                        print(f"[DEBUG UI] Traceback: {traceback.format_exc()}")
                        continue
            
            print(f"[DEBUG UI] Final mapping: {mapping}")
            
            # Call preview update endpoint
            r = requests.post(
                f"{API}/preview_correction_update/{run_id_val}/{absolute_frame}",
                json={"mapping": mapping},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            
            # Decode image
            import numpy as np, cv2
            import base64
            image_data = data["image"].split(",")[1]
            img_bytes = base64.b64decode(image_data)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            print(f"[DEBUG UI] Successfully updated preview image, shape: {im.shape}")
            # Return both the image and ensure it's visible
            return gr.update(value=im, visible=True)
        except Exception as e:
            print(f"[DEBUG UI] Failed to update preview: {e}")
            import traceback
            print(f"[DEBUG UI] Traceback: {traceback.format_exc()}")
            # Return no update on error - keep current image
            return gr.update()
    
    # Apply correction - saves with user's ID mapping
    def apply_correction_wrapper(run_id_val, absolute_frame, id_mapping_df):
        """Apply user's ID mapping and save corrected frame."""
        if not run_id_val:
            raise gr.Error("No run_id yet.")
        
        if absolute_frame is None:
            raise gr.Error("No frame selected. Please prepare correction first.")
        
        if id_mapping_df is None or len(id_mapping_df) == 0:
            raise gr.Error("No ID mappings provided.")
        
        # Convert dataframe to dict: mask_index -> final_id
        # Gradio Dataframe returns pandas DataFrame
        id_mapping = {}
        print(f"[DEBUG UI] id_mapping_df type: {type(id_mapping_df)}")
        print(f"[DEBUG UI] id_mapping_df:\n{id_mapping_df}")
        
        # Handle pandas DataFrame
        import pandas as pd
        if isinstance(id_mapping_df, pd.DataFrame):
            # DataFrame has columns: "Mask #", "Auto-assigned ID", "Final ID", "Delete?"
            for idx, row in id_mapping_df.iterrows():
                try:
                    mask_idx = int(row["Mask #"])
                    delete = row.get("Delete?", False)
                    
                    # Skip if marked for deletion
                    if delete:
                        print(f"[DEBUG UI] Row {idx}: mask {mask_idx} marked for deletion, skipping")
                        continue
                    
                    final_id = row["Final ID"]
                    if pd.isna(final_id) or final_id == "":
                        raise gr.Error(f"Final ID is required for mask {mask_idx} (or mark as Delete).")
                    
                    final_id = int(final_id)
                    if final_id < 1:
                        raise gr.Error(f"Invalid ID {final_id} for mask {mask_idx}. IDs must be >= 1.")
                    
                    id_mapping[str(mask_idx)] = final_id
                    print(f"[DEBUG UI] Row {idx}: mask {mask_idx} -> ID {final_id}")
                except (ValueError, TypeError, KeyError) as e:
                    print(f"[DEBUG UI] Error parsing row {idx}: {row}, error: {e}")
                    raise gr.Error(f"Error parsing ID mapping row {idx+1}. Please check the format.")
        else:
            # Fallback: treat as list of lists
            for i, row in enumerate(id_mapping_df):
                try:
                    mask_idx = int(float(row[0]))
                    final_id = int(float(row[2]))
                    
                    if final_id < 1:
                        raise gr.Error(f"Invalid ID {final_id} for mask {mask_idx}. IDs must be >= 1.")
                    
                    id_mapping[str(mask_idx)] = final_id
                except (ValueError, TypeError, IndexError) as e:
                    raise gr.Error(f"Error parsing ID mapping row {i+1}: {row}. Please check the format.")
        
        print(f"[DEBUG] Applying correction with mapping: {id_mapping}")
        
        # Call apply_correction endpoint
        r = requests.post(
            f"{API}/apply_correction/{run_id_val}/{absolute_frame}",
            json={"mapping": id_mapping},
            timeout=3600,
        )
        r.raise_for_status()
        result = r.json()
        
        # Get updated golden video and progress
        golden_path = _download_to_temp(f"{API}/golden_video/{run_id_val}", suffix=".mp4")
        prog = _get_progress(run_id_val)
        
        status_msg = f"✅ **Frame {absolute_frame} corrected and saved!**\n"
        status_msg += f"- Applied ID mapping: {id_mapping}\n"
        status_msg += f"- Max ID: {result['max_id']}\n"
        status_msg += f"{_progress_text(prog)}"
        
        return (
            None,  # corrected_frame (don't show image)
            golden_path,  # golden_video
            status_msg,  # correction_status
            float(prog.get("golden_percent", 0.0)),  # progress_bar
            gr.update(visible=False),  # corrected_frame (hide it)
            gr.update(visible=False),  # id_reassignment_section (hide after applying)
        )
    
    # Store absolute frame for apply step
    absolute_frame_storage = gr.State(value=None)
    # Store image dimensions for coordinate scaling
    correction_image_dims_storage = gr.State(value=None)  # (width, height)
    
    # Get frame preview (to find wrong frame)
    def get_frame_preview(run_id_val, frame_idx):
        """Get and display a frame from the tracked chunk."""
        print(f"[DEBUG UI] get_frame_preview: run_id={run_id_val}, frame_idx={frame_idx}")
        
        if not run_id_val:
            return (
                None,
                gr.update(visible=False),
                "❌ No run_id yet.",
                None,
            )
        
        if frame_idx is None or frame_idx == "":
            return (
                None,
                gr.update(visible=False),
                "❌ Please enter a frame number.",
                None,
            )
        
        try:
            relative_frame = int(frame_idx)
            if relative_frame < 0:
                return (
                    None,
                    gr.update(visible=False),
                    "❌ Frame number must be >= 0.",
                    None,
                )
            
            print(f"[DEBUG UI] Fetching frame {relative_frame} from backend...")
            
            # Call backend to get frame
            r = requests.get(
                f"{API}/tracked_frame/{run_id_val}/{relative_frame}",
                timeout=60,
            )
            r.raise_for_status()
            
            # Decode image
            import numpy as np, cv2
            arr = np.frombuffer(r.content, dtype=np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            # Get absolute frame for info
            prog = _get_progress(run_id_val)
            max_idx = prog.get("golden_max_idx")
            if max_idx is not None:
                absolute_frame = int(max_idx) + relative_frame
                info = f"✅ **Frame {relative_frame}** (absolute: {absolute_frame})"
            else:
                info = f"✅ **Frame {relative_frame}**"
            
            print(f"[DEBUG UI] Successfully got frame {relative_frame}")
            
            return (
                im,
                gr.update(visible=True),
                info,
                relative_frame,  # Auto-fill correction input
            )
        except requests.exceptions.HTTPError as e:
            error_msg = f"❌ Error: {e.response.json().get('detail', str(e))}"
            print(f"[DEBUG UI] HTTP Error: {e}")
            return (
                None,  # correction_preview_frame
                None,  # id_mapping_table
                error_msg,  # correction_status
                gr.update(visible=False),  # id_reassignment_section
                gr.update(visible=False),  # correction_preview_frame
                None,  # absolute_frame_storage
                None,  # golden_video
                error_msg,  # progress_md
                None,  # progress_bar
            )
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(f"[DEBUG UI] Error: {e}")
            import traceback
            print(f"[DEBUG UI] Traceback: {traceback.format_exc()}")
            return (
                None,  # correction_preview_frame
                None,  # id_mapping_table
                error_msg,  # correction_status
                gr.update(visible=False),  # id_reassignment_section
                gr.update(visible=False),  # correction_preview_frame
                None,  # absolute_frame_storage
                None,  # golden_video
                error_msg,  # progress_md
                None,  # progress_bar
            )
    
    # Wire up preview frame button
    get_frame_btn.click(
        get_frame_preview,
        [run_id, preview_frame_idx],
        [previewed_frame, previewed_frame, correction_status, wrong_frame_idx],
    )
    
    # Wire up prepare correction button
    prepare_correction_btn.click(
        prepare_correction_wrapper,
        [run_id, wrong_frame_idx],
        [correction_preview_frame, id_mapping_table, correction_status, id_reassignment_section, correction_preview_frame, absolute_frame_storage, correction_image_dims_storage, golden_video, progress_md, progress_bar],
    )
    
    # Handle clicks on correction preview for mask refinement
    def handle_mask_refinement_click(run_id_val, absolute_frame, image_dims, selected_mask_idx, point_mode_val, evt: gr.SelectData):
        """Handle click on correction preview image to refine mask."""
        if not run_id_val or absolute_frame is None:
            return gr.update(), gr.update(visible=False, value="❌ No run_id or frame selected. Please prepare correction first.")
        
        if selected_mask_idx is None:
            return gr.update(), gr.update(visible=True, value="❌ Please select a mask number first (0, 1, 2, ...)")
        
        try:
            mask_idx = int(selected_mask_idx)
            
            # Gradio SelectData provides coordinates in the displayed image space
            # We need to scale them to the actual image pixel space
            # evt.index is (row, col) = (y, x) in the displayed image
            displayed_y, displayed_x = evt.index[0], evt.index[1]
            
            # Get actual image dimensions for scaling
            if image_dims is not None:
                actual_width, actual_height = image_dims
                
                # Try to get displayed image size from evt.value if available
                # Gradio might provide the displayed size in evt.value
                displayed_width = None
                displayed_height = None
                
                if hasattr(evt, 'value') and evt.value is not None:
                    # evt.value might contain image info
                    try:
                        if isinstance(evt.value, dict):
                            displayed_width = evt.value.get('width')
                            displayed_height = evt.value.get('height')
                    except:
                        pass
                
                # If we have displayed dimensions, scale coordinates
                if displayed_width and displayed_height and displayed_width > 0 and displayed_height > 0:
                    scale_x = actual_width / displayed_width
                    scale_y = actual_height / displayed_height
                    x = int(displayed_x * scale_x)
                    y = int(displayed_y * scale_y)
                    print(f"[DEBUG UI] Scaled coordinates: displayed=({displayed_x}, {displayed_y}), scale=({scale_x:.3f}, {scale_y:.3f}), actual=({x}, {y})")
                else:
                    # Fallback: assume coordinates are already in actual image space
                    # (This might be the case for numpy images)
                    x, y = int(displayed_x), int(displayed_y)
                    print(f"[DEBUG UI] Using coordinates directly (no displayed size info): ({x}, {y})")
                
                # Validate coordinates are within image bounds
                if x < 0 or x >= actual_width or y < 0 or y >= actual_height:
                    print(f"[DEBUG UI] Warning: Scaled coordinates ({x}, {y}) outside image bounds ({actual_width}, {actual_height})")
                    # Clamp to valid range
                    x = max(0, min(x, actual_width - 1))
                    y = max(0, min(y, actual_height - 1))
                    print(f"[DEBUG UI] Clamped to: ({x}, {y})")
            else:
                # No image dimensions available, use coordinates directly
                x, y = int(displayed_x), int(displayed_y)
                print(f"[DEBUG UI] Warning: image_dims not available, using displayed coordinates directly: ({x}, {y})")
            
            is_positive = point_mode_val == "Add points"
            
            print(f"[DEBUG UI] Mask refinement click: mask={mask_idx}, displayed=({displayed_x}, {displayed_y}), final=({x}, {y}), is_positive={is_positive}")
            if image_dims:
                print(f"[DEBUG UI] Image dimensions: {image_dims}")
            
            # Call refine_mask endpoint
            r = requests.post(
                f"{API}/refine_mask/{run_id_val}/{absolute_frame}",
                json={
                    "mask_index": mask_idx,
                    "points": [{"x": int(x), "y": int(y), "is_positive": is_positive}],
                },
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            
            # Decode updated preview image
            import numpy as np, cv2
            import base64
            image_data = data["image"].split(",")[1]
            img_bytes = base64.b64decode(image_data)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            # Update stored image dimensions if they changed
            returned_width = data.get('image_width')
            returned_height = data.get('image_height')
            if returned_width and returned_height:
                new_dims = (returned_width, returned_height)
            else:
                new_dims = image_dims  # Keep old dimensions if not provided
            
            refined_size = data.get('refined_mask_size', 'unknown')
            mode_text = "added" if is_positive else "removed"
            status_msg = f"✅ Mask {mask_idx} refined! Point ({x}, {y}) {mode_text}. New mask size: {refined_size} pixels."
            print(f"[DEBUG UI] Mask {mask_idx} refined, new size: {refined_size} pixels, point used: ({x}, {y})")
            
            return (
                gr.update(value=im, visible=True),  # Updated preview image
                gr.update(visible=True, value=status_msg),  # Status message
                new_dims,  # Update image dimensions
            )
        except Exception as e:
            print(f"[DEBUG UI] Error refining mask: {e}")
            import traceback
            print(f"[DEBUG UI] Traceback: {traceback.format_exc()}")
            error_msg = f"❌ Failed to refine mask: {str(e)}"
            return gr.update(), gr.update(visible=True, value=error_msg), image_dims  # Keep dimensions on error
    
    # Wire up table to update preview in real-time
    id_mapping_table.change(
        update_preview_from_table,
        [run_id, absolute_frame_storage, id_mapping_table],
        [correction_preview_frame],
    )
    
    # Wire up click handler for mask refinement on correction preview image
    correction_preview_frame.select(
        handle_mask_refinement_click,
        [run_id, absolute_frame_storage, correction_image_dims_storage, selected_mask_idx, point_mode],
        [correction_preview_frame, refinement_status, correction_image_dims_storage],
    )
    
    # Wire up apply correction button
    apply_correction_btn.click(
        apply_correction_wrapper,
        [run_id, absolute_frame_storage, id_mapping_table],
        [corrected_frame, golden_video, correction_status, progress_bar, corrected_frame, id_reassignment_section],
    )
    
    # Cancel button - hide the reassignment section
    cancel_correction_btn.click(
        lambda: gr.update(visible=False),
        None,
        [id_reassignment_section],
    )

demo.launch(css=CSS)
