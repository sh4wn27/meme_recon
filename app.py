import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import logging
import time
import json

# Suppress MediaPipe verbose logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# ---------- Helpers ----------
def dist(a, b):
    return np.linalg.norm(a - b)

def safe_ratio(num, den, eps=1e-6):
    return float(num) / float(den + eps)

def extract_expression_features(pts):
    """
    pts: (468,2) array of face landmarks in pixel coords.
    Returns a small feature vector describing expression.
    """

    # A few useful landmark indices (MediaPipe Face Mesh)
    # Mouth: left/right corners, top/bottom lip
    mouth_left  = pts[61]
    mouth_right = pts[291]
    mouth_top   = pts[13]
    mouth_bot   = pts[14]

    # Eyes: top/bottom lids (approx)
    left_eye_top  = pts[159]
    left_eye_bot  = pts[145]
    right_eye_top = pts[386]
    right_eye_bot = pts[374]

    # Eyebrows: inner/outer points (approx)
    left_brow = pts[105]
    right_brow = pts[334]

    # Nose bridge / reference scale
    nose_tip = pts[1]
    chin = pts[152]
    face_scale = dist(nose_tip, chin)  # normalize distances by face size

    # Metrics
    mouth_width = safe_ratio(dist(mouth_left, mouth_right), face_scale)
    mouth_open  = safe_ratio(dist(mouth_top, mouth_bot), face_scale)

    left_eye_open  = safe_ratio(dist(left_eye_top, left_eye_bot), face_scale)
    right_eye_open = safe_ratio(dist(right_eye_top, right_eye_bot), face_scale)
    eye_open = (left_eye_open + right_eye_open) / 2.0

    # Eyebrow height proxy: brow y relative to nose tip y (in image coords y increases downward)
    brow_raise = safe_ratio((nose_tip[1] - (left_brow[1] + right_brow[1]) / 2.0), face_scale)

    # Feature vector: tweak/extend as you like
    # [smile-ish width, mouth open, eye open, brow raise]
    return np.array([mouth_width, mouth_open, eye_open, brow_raise], dtype=np.float32)

def load_memes(meme_dir="memes"):
    # Map expression label -> file
    mapping = {
        "look": "cat_look.png",
        "pondering": "cat_pondering.png",
        "tongue": "cat_tongue.png",
        "enlightment": "monkey_enlightment.png",
        "thinking": "monkey_thinkning.png",
    }
    memes = {}
    for label, fname in mapping.items():
        path = os.path.join(meme_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            memes[label] = img
    return memes

# Default targets (used if no saved targets exist)
DEFAULT_TARGETS = {
    "look":     np.array([0.38, 0.02, 0.03, 0.02], dtype=np.float32),
    "pondering": np.array([0.32, 0.09, 0.05, 0.05], dtype=np.float32),
    "tongue":     np.array([0.30, 0.01, 0.03, 0.00], dtype=np.float32),
    "enlightment":    np.array([0.30, 0.02, 0.015, 0.01], dtype=np.float32),
    "thinking":    np.array([0.30, 0.02, 0.015, 0.01], dtype=np.float32),
}

TARGETS_FILE = "expression_targets.json"

def load_targets():
    """Load saved targets from file, or return defaults if file doesn't exist."""
    if os.path.exists(TARGETS_FILE):
        try:
            with open(TARGETS_FILE, 'r') as f:
                data = json.load(f)
                # Handle both old format (just arrays) and new format (dicts with face/hand)
                targets = {}
                for k, v in data.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                        # Old format: just face features array
                        targets[k] = {
                            'face': np.array(v, dtype=np.float32),
                            'hand_gesture': None  # No hand gesture
                        }
                    elif isinstance(v, dict):
                        # New format: dict with face and optional hand_gesture
                        targets[k] = {
                            'face': np.array(v.get('face', v), dtype=np.float32),
                            'hand_gesture': v.get('hand_gesture', None)  # Can be None or a gesture string
                        }
                    else:
                        # Fallback
                        targets[k] = {
                            'face': np.array(v, dtype=np.float32),
                            'hand_gesture': None
                        }
                print(f"Loaded {len(targets)} saved expression targets from {TARGETS_FILE}")
                return targets
        except Exception as e:
            print(f"Error loading targets: {e}. Using defaults.")
            # Convert defaults to new format
            return {k: {'face': v, 'hand_gesture': None} for k, v in DEFAULT_TARGETS.items()}
    # Convert defaults to new format
    return {k: {'face': v, 'hand_gesture': None} for k, v in DEFAULT_TARGETS.items()}

def save_targets(targets):
    """Save targets to file."""
    # Convert to JSON-serializable format
    data = {}
    for k, v in targets.items():
        if isinstance(v, dict):
            data[k] = {
                'face': v['face'].tolist() if isinstance(v['face'], np.ndarray) else v['face'],
                'hand_gesture': v.get('hand_gesture', None)
            }
        else:
            # Handle old format if needed
            data[k] = {
                'face': v.tolist() if isinstance(v, np.ndarray) else v,
                'hand_gesture': None
            }
    with open(TARGETS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(targets)} expression targets to {TARGETS_FILE}")

# Load targets (will use defaults if file doesn't exist)
TARGETS = load_targets()

def predict_label(feat, detected_gesture=None, threshold=0.08):
    """
    Predict which expression matches the features.
    If a hand gesture is required for a target, both face AND gesture must match.
    Only returns a match if the score is below the threshold.
    Returns (label, score) or (None, score) if no good match found.
    
    Args:
        feat: Face feature vector
        detected_gesture: Detected hand gesture string (or None if no hand detected)
        threshold: Matching threshold (higher = more lenient)
    """
    best_label, best_score = None, 1e9
    for label, target_data in TARGETS.items():
        # Handle both old format (just array) and new format (dict)
        if isinstance(target_data, dict):
            tgt_face = target_data['face']
            required_gesture = target_data.get('hand_gesture', None)
        else:
            # Old format compatibility
            tgt_face = target_data
            required_gesture = None
        
        # Calculate face match score
        face_score = np.linalg.norm(feat - tgt_face)
        
        # Check if hand gesture is required and matches
        hand_match = True
        if required_gesture is not None:
            # This expression requires a hand gesture
            if detected_gesture is None or detected_gesture != required_gesture:
                # Hand gesture missing or doesn't match - skip this target
                continue
            hand_match = True
        
        # Use face score for ranking
        if face_score < best_score:
            best_score = face_score
            best_label = label
    
    # Only return a match if it's close enough
    if best_score > threshold:
        return None, best_score
    return best_label, best_score

def overlay_meme(frame, meme_img, scale=0.35):
    if meme_img is None:
        return frame
    h, w = frame.shape[:2]
    mh, mw = meme_img.shape[:2]
    new_w = int(w * scale)
    new_h = int(mh * (new_w / mw))
    meme_resized = cv2.resize(meme_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # top-right corner
    x1 = w - new_w - 10
    y1 = 10
    x2 = x1 + new_w
    y2 = y1 + new_h

    # bounds
    if x1 < 0 or y2 > h:
        return frame

    frame[y1:y2, x1:x2] = meme_resized
    return frame

# ---------- Hand Detection Helpers ----------
def draw_hand_landmarks(frame, hand_landmarks, hand_label="Left"):
    """Draw hand landmarks and connections on the frame."""
    h, w = frame.shape[:2]
    
    # Hand landmark connections (simplified - showing key points)
    # MediaPipe hand has 21 landmarks
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17)
    ]
    
    # Draw connections
    color = (0, 255, 0) if hand_label == "Left" else (255, 0, 0)
    for start_idx, end_idx in connections:
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        start_pt = (int(start.x * w), int(start.y * h))
        end_pt = (int(end.x * w), int(end.y * h))
        cv2.line(frame, start_pt, end_pt, color, 2)
    
    # Draw landmarks
    for landmark in hand_landmarks:
        pt = (int(landmark.x * w), int(landmark.y * h))
        cv2.circle(frame, pt, 5, color, -1)
    
    return frame

def detect_gesture(hand_landmarks):
    """
    Detect simple gestures from hand landmarks.
    Returns: 'pointing', 'thumbs_up', 'fist', 'open_hand', 'peace', or None
    """
    # Convert to pixel coordinates (normalized 0-1)
    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
    
    # Key points (MediaPipe hand has 21 landmarks)
    # 0: wrist, 4: thumb tip, 8: index tip, 12: middle tip, 16: ring tip, 20: pinky tip
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # MCP joints (base of fingers)
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]
    
    # Check if fingers are extended (tip is above MCP joint in y-coordinate)
    def is_finger_extended(tip, mcp):
        return tip[1] < mcp[1]  # Lower y = higher on screen
    
    index_extended = is_finger_extended(index_tip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_mcp)
    
    # Thumb is special - check if it's extended outward
    thumb_extended = thumb_tip[0] > landmarks[3][0]  # Thumb tip is to the right of joint
    
    # Gesture detection
    extended_count = sum([index_extended, middle_extended, ring_extended, pinky_extended])
    
    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "pointing"
    elif thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "thumbs_up"
    elif index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "peace"
    elif extended_count == 4:
        return "open_hand"
    elif extended_count == 0:
        return "fist"
    
    return None

# ---------- Calibration Mode ----------
def calibrate_expressions(camera_index=None, samples_per_expression=10):
    """
    Calibration mode: Capture your expressions to train the model.
    For each expression, make the face and press SPACEBAR to capture samples.
    Press 's' to save and move to next expression, 'q' to quit.
    """
    print("=" * 60)
    print("CALIBRATION MODE")
    print("=" * 60)
    print(f"Will capture {samples_per_expression} samples per expression")
    print("Instructions:")
    print("  1. Make the expression shown on screen (with your face)")
    print("  2. OPTIONAL: Make a hand gesture (pointing, thumbs_up, peace, etc.)")
    print("     - If you make a gesture in >50% of samples, it will be required for matching")
    print("     - Expressions with gestures require BOTH face AND gesture to match")
    print("  3. Press SPACEBAR to capture a sample (do this multiple times)")
    print("  4. Press 's' to save this expression and move to next")
    print("  5. Press 'r' to restart/redo current expression (clear samples)")
    print("  6. Press 'k' to skip this expression and move to next")
    print("  7. Press 'd' to delete this expression's calibration (if exists)")
    print("  8. Press 'q' to quit calibration")
    print("=" * 60)
    
    # Get available expressions from memes
    memes = load_memes("meme-recon/memes")
    expression_labels = list(memes.keys())
    
    if not expression_labels:
        print("ERROR: No memes found! Cannot calibrate.")
        return
    
    # Check which expressions are already calibrated
    existing_targets = load_targets()
    calibrated = [label for label in expression_labels if label in existing_targets]
    not_calibrated = [label for label in expression_labels if label not in existing_targets]
    
    print(f"\nExpressions to calibrate: {', '.join(expression_labels)}")
    if calibrated:
        print(f"Already calibrated: {', '.join(calibrated)} (will be overwritten if you calibrate again)")
    if not_calibrated:
        print(f"Not yet calibrated: {', '.join(not_calibrated)}")
    
    input("\nPress ENTER to start calibration...")
    
    # Setup camera
    print("\nOpening webcam...")
    cap = None
    
    if camera_index is not None:
        # Use specified camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"ERROR: Could not open camera {camera_index}")
            return
        print(f"Using camera index {camera_index}")
    else:
        # Try camera 1 first (to avoid the "plank of food" camera 0)
        for camera_idx in [1, 0]:
            print(f"Trying camera {camera_idx}...")
            test_cap = cv2.VideoCapture(camera_idx)
            # Give it a moment to initialize
            time.sleep(0.1)
            
            if test_cap.isOpened():
                # Try to read a few frames to make sure it's actually working
                ret = False
                for _ in range(5):  # Try reading up to 5 times
                    ret, test_frame = test_cap.read()
                    if ret and test_frame is not None:
                        print(f"✓ Camera {camera_idx} is working!")
                        cap = test_cap
                        break
                    time.sleep(0.1)
                
                if cap is not None:
                    break
                else:
                    print(f"✗ Camera {camera_idx} opened but couldn't read frames")
                    test_cap.release()
            else:
                print(f"✗ Camera {camera_idx} failed to open")
                if test_cap.isOpened():
                    test_cap.release()
        
        if cap is None or not cap.isOpened():
            print("\nERROR: Could not find a working webcam.")
            print("Available cameras: 0, 1")
            print("Note: Camera 0 might show the wrong feed. Try: calibrate_expressions(camera_index=1)")
            return
    
    # Setup face detector
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_detector = vision.FaceLandmarker.create_from_options(options)
    
    # Setup hand detector
    print("Initializing hand detector...")
    hand_model_path = "hand_landmarker.task"
    if not os.path.exists(hand_model_path):
        print(f"WARNING: Hand model file not found: {hand_model_path}")
        print("Hand detection will be disabled. Continuing with face only...")
        hand_detector = None
    else:
        hand_base_options = python.BaseOptions(model_asset_path=hand_model_path)
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        print("Hand detector ready!")
    
    # Calibrate each expression
    new_targets = {}
    deleted_expressions = set()  # Track which expressions to delete
    # Load existing targets to check what's already calibrated
    existing_targets = load_targets()
    
    for expr_label in expression_labels:
        # Check if already calibrated
        is_already_calibrated = expr_label in existing_targets
        
        print(f"\n{'='*60}")
        print(f"CALIBRATING: {expr_label.upper()}")
        if is_already_calibrated:
            print(f"⚠️  This expression is already calibrated (will be overwritten)")
        print(f"{'='*60}")
        print(f"Make the '{expr_label}' expression and press SPACEBAR {samples_per_expression} times")
        print("Press 's' when done, 'r' to restart, 'k' to skip, 'd' to delete, 'q' to quit")
        
        samples = []
        hand_gestures = []  # Track hand gestures during calibration
        saved = False
        skipped = False
        
        while not saved:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # Detect face
            face_res = face_detector.detect(mp_image)
            
            # Detect hands
            hand_res = None
            if hand_detector:
                hand_res = hand_detector.detect(mp_image)
            
            # Draw instructions
            status_text = f"Expression: {expr_label} | Samples: {len(samples)}/{samples_per_expression}"
            if is_already_calibrated:
                status_text += " [ALREADY CALIBRATED]"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE=capture | s=save | r=restart | k=skip | d=delete | q=quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw and process hands
            if hand_res and hand_res.hand_landmarks:
                for idx, hand_landmarks in enumerate(hand_res.hand_landmarks):
                    hand_label = hand_res.handedness[idx][0].category_name if idx < len(hand_res.handedness) else "Hand"
                    frame = draw_hand_landmarks(frame, hand_landmarks, hand_label)
                    
                    # Show gesture
                    gesture = detect_gesture(hand_landmarks)
                    if gesture:
                        gesture_text = f"{hand_label}: {gesture}"
                        cv2.putText(frame, gesture_text, (10, 120 + idx * 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if face_res.face_landmarks:
                h, w = frame.shape[:2]
                lm = face_res.face_landmarks[0]
                pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
                feat = extract_expression_features(pts)
                
                # Show current features
                feat_text = f"Features: [{feat[0]:.3f}, {feat[1]:.3f}, {feat[2]:.3f}, {feat[3]:.3f}]"
                y_offset = 90 if not (hand_res and hand_res.hand_landmarks) else 150
                cv2.putText(frame, feat_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Draw face landmarks (optional, for visual feedback)
                for pt in pts[::10]:  # Show every 10th point
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
            else:
                y_offset = 90 if not (hand_res and hand_res.hand_landmarks) else 150
                cv2.putText(frame, "No face detected!", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Calibration Mode", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACEBAR
                if face_res.face_landmarks:
                    h, w = frame.shape[:2]
                    lm = face_res.face_landmarks[0]
                    pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
                    feat = extract_expression_features(pts)
                    samples.append(feat)
                    
                    # Also capture hand gesture if present
                    detected_gesture = None
                    if hand_res and hand_res.hand_landmarks:
                        # Use the first hand's gesture
                        gesture = detect_gesture(hand_res.hand_landmarks[0])
                        if gesture:
                            detected_gesture = gesture
                            hand_gestures.append(gesture)
                    
                    print(f"  Captured sample {len(samples)}/{samples_per_expression}", end="")
                    if detected_gesture:
                        print(f" (gesture: {detected_gesture})")
                    else:
                        print()
                    if len(samples) >= samples_per_expression:
                        print(f"  ✓ Collected {len(samples)} samples!")
                else:
                    print("  ✗ No face detected, cannot capture")
            
            elif key == ord('s'):  # Save
                if len(samples) >= 3:  # Need at least 3 samples
                    # Average the samples
                    avg_feat = np.mean(samples, axis=0)
                    
                    # Determine most common hand gesture (if any)
                    saved_gesture = None
                    if hand_gestures:
                        # Find most common gesture
                        from collections import Counter
                        gesture_counts = Counter(hand_gestures)
                        most_common = gesture_counts.most_common(1)[0]
                        # Only save gesture if it appears in at least 50% of samples
                        if most_common[1] >= len(samples) * 0.5:
                            saved_gesture = most_common[0]
                            print(f"  ✓ Detected hand gesture: {saved_gesture} (in {most_common[1]}/{len(samples)} samples)")
                    
                    # Save with new format
                    new_targets[expr_label] = {
                        'face': avg_feat,
                        'hand_gesture': saved_gesture
                    }
                    
                    gesture_text = f" with gesture: {saved_gesture}" if saved_gesture else " (face only)"
                    print(f"  ✓ Saved {expr_label}{gesture_text}")
                    print(f"     Features: {avg_feat}")
                    saved = True
                else:
                    print(f"  ✗ Need at least 3 samples (have {len(samples)})")
            
            elif key == ord('r'):  # Restart/Redo
                samples = []
                hand_gestures = []
                print(f"  ↻ Restarted {expr_label} - samples cleared")
            
            elif key == ord('k'):  # Skip
                print(f"  ⏭️  Skipped {expr_label}")
                skipped = True
                saved = True  # Exit loop
            
            elif key == ord('d'):  # Delete
                if expr_label in existing_targets or expr_label in new_targets:
                    # Mark for deletion
                    deleted_expressions.add(expr_label)
                    # Remove from new_targets if present
                    if expr_label in new_targets:
                        del new_targets[expr_label]
                    print(f"  🗑️  Marked {expr_label} for deletion")
                    print(f"     (Will be removed when you save. Press 'k' to skip or calibrate again)")
                else:
                    print(f"  ℹ️  {expr_label} is not calibrated, nothing to delete")
            
            elif key == ord('q'):  # Quit
                print("\nCalibration cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # Show meme for this expression (only if not skipped)
        if not skipped and expr_label in memes:
            meme_img = memes[expr_label]
            h, w = frame.shape[:2]
            mh, mw = meme_img.shape[:2]
            scale = min(w * 0.3 / mw, h * 0.3 / mh)
            meme_resized = cv2.resize(meme_img, (int(mw * scale), int(mh * scale)))
            frame[10:10+meme_resized.shape[0], 10:10+meme_resized.shape[1]] = meme_resized
            cv2.imshow("Calibration Mode", frame)
            print("  This is the meme for this expression!")
            time.sleep(2)
    
    # Merge new targets with existing, remove deleted ones
    final_targets = {}
    
    # Start with existing targets
    for label, target in existing_targets.items():
        if label not in deleted_expressions:
            final_targets[label] = target
    
    # Add/overwrite with new calibrations
    for label, target in new_targets.items():
        final_targets[label] = target
    
    # Save all targets (merged)
    if final_targets or deleted_expressions:
        save_targets(final_targets)
        print(f"\n{'='*60}")
        print("CALIBRATION COMPLETE!")
        print(f"{'='*60}")
        if new_targets:
            print(f"Saved/updated {len(new_targets)} expression target(s).")
        if deleted_expressions:
            print(f"Deleted {len(deleted_expressions)} expression(s): {', '.join(deleted_expressions)}")
        print(f"Total calibrated expressions: {len(final_targets)}")
        print("You can now run the main app and it will use your calibrated expressions!")
    else:
        print("\nNo changes made to calibrations.")
    
    cap.release()
    cv2.destroyAllWindows()

# ---------- Main ----------
def main(camera_index=None):
    """
    camera_index: Which camera to use (0, 1, 2, etc.). If None, will try to find one automatically.
    """
    # Reload targets in case they were updated
    global TARGETS
    TARGETS = load_targets()
    
    print("Loading memes...")
    memes = load_memes("meme-recon/memes")
    print(f"Loaded {len(memes)} memes")
    
    print("Opening webcam...")
    cap = None
    
    if camera_index is not None:
        # Use specified camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"ERROR: Could not open camera {camera_index}")
            raise RuntimeError(f"Could not open camera {camera_index}")
        print(f"Using camera index {camera_index}")
    else:
        # Try camera 1 first (to avoid the "plank of food" camera 0)
        # Only try 0 and 1 since those are the available cameras
        for camera_idx in [1, 0]:
            print(f"Trying camera {camera_idx}...")
            test_cap = cv2.VideoCapture(camera_idx)
            # Give it a moment to initialize
            time.sleep(0.1)
            
            if test_cap.isOpened():
                # Try to read a few frames to make sure it's actually working
                ret = False
                for _ in range(5):  # Try reading up to 5 times
                    ret, test_frame = test_cap.read()
                    if ret and test_frame is not None:
                        print(f"✓ Camera {camera_idx} is working!")
                        cap = test_cap
                        break
                    time.sleep(0.1)
                
                if cap is not None:
                    break
                else:
                    print(f"✗ Camera {camera_idx} opened but couldn't read frames")
                    test_cap.release()
            else:
                print(f"✗ Camera {camera_idx} failed to open")
                if test_cap.isOpened():
                    test_cap.release()
        
        if cap is None or not cap.isOpened():
            print("\nERROR: Could not find a working webcam.")
            print("Available cameras: 0, 1")
            print("Note: Camera 0 might show the wrong feed. Try: main(camera_index=1)")
            raise RuntimeError("Could not find a working webcam.")
    
    print("Initializing face detector...")

    # Initialize FaceLandmarker with the new API
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please download it.")
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_detector = vision.FaceLandmarker.create_from_options(options)
    
    # Initialize HandLandmarker
    print("Initializing hand detector...")
    hand_model_path = "hand_landmarker.task"
    if not os.path.exists(hand_model_path):
        raise FileNotFoundError(f"Hand model file not found: {hand_model_path}. Please download it.")
    
    hand_base_options = python.BaseOptions(model_asset_path=hand_model_path)
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        num_hands=2,  # Detect up to 2 hands
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    
    print("Face and hand detectors ready! Starting video capture...")
    print("Press 'q' or ESC to quit")
    print("Look for a window titled 'Meme Expression Matcher'")

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam")
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print every 30 frames
            print(f"Processing frame {frame_count}...")

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Detect face
        face_res = face_detector.detect(mp_image)
        
        # Detect hands
        hand_res = hand_detector.detect(mp_image)

        label = "no_face"
        score = None

        # Detect hand gesture first (needed for prediction)
        detected_gesture = None
        if hand_res.hand_landmarks:
            # Use the first detected hand's gesture
            gesture = detect_gesture(hand_res.hand_landmarks[0])
            if gesture:
                detected_gesture = gesture

        # Process face detection
        if face_res.face_landmarks:
            h, w = frame.shape[:2]
            lm = face_res.face_landmarks[0]
            pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)

            feat = extract_expression_features(pts)
            # threshold=0.20 means show meme if expression matches (more lenient)
            # Lower threshold = stricter matching, Higher = more lenient
            # Pass detected gesture - if expression requires a gesture, both must match
            label, score = predict_label(feat, detected_gesture=detected_gesture, threshold=0.08)

            # Only overlay meme if we have a good match
            if label is not None:
                # overlay meme if we have it
                frame = overlay_meme(frame, memes.get(label))
            else:
                # No good match found
                label = "no_match"

        # Process and draw hands
        gesture_text = ""
        if hand_res.hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_res.hand_landmarks):
                # Get hand label (Left or Right)
                hand_label = hand_res.handedness[idx][0].category_name if idx < len(hand_res.handedness) else "Hand"
                
                # Draw hand landmarks
                frame = draw_hand_landmarks(frame, hand_landmarks, hand_label)
                
                # Detect gesture
                gesture = detect_gesture(hand_landmarks)
                if gesture:
                    gesture_text = f"{hand_label}: {gesture}"
                    # Draw gesture text
                    cv2.putText(frame, gesture_text, (10, 70 + idx * 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display face expression label
        txt = f"{label}" if score is None else f"{label}  (score={score:.3f})"
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        # Show gesture requirement if expression requires one
        if label and label != "no_face" and label != "no_match":
            if label in TARGETS:
                target_data = TARGETS[label]
                if isinstance(target_data, dict) and target_data.get('hand_gesture'):
                    required_gesture = target_data['hand_gesture']
                    gesture_status = "✓" if detected_gesture == required_gesture else "✗"
                    gesture_txt = f"Gesture: {gesture_status} {required_gesture}"
                    color = (0, 255, 0) if detected_gesture == required_gesture else (0, 0, 255)
                    cv2.putText(frame, gesture_txt, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Meme Expression Matcher", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "calibrate":
            print("Starting calibration mode...")
            calibrate_expressions()
        elif command in ["clear", "reset", "delete"]:
            # Delete all calibrations
            if os.path.exists(TARGETS_FILE):
                os.remove(TARGETS_FILE)
                print(f"✓ Deleted all calibrations ({TARGETS_FILE})")
                print("You can now run 'python app.py calibrate' to start fresh!")
            else:
                print(f"No calibration file found ({TARGETS_FILE})")
                print("Nothing to delete.")
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  python app.py          - Run the app")
            print("  python app.py calibrate - Calibrate expressions")
            print("  python app.py clear    - Delete all calibrations")
    else:
        # Normal mode
        # To use a specific camera, pass the index: main(camera_index=1)
        # The code will auto-detect and skip camera 0 (the "plank of food" one)
        print("Starting meme recognition app...")
        print("To calibrate your expressions, run: python app.py calibrate")
        print("To delete all calibrations, run: python app.py clear")
        main()
