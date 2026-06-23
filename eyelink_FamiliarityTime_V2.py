"""
EyeLink 1000 Face Detection Experiment
Using PsychoPy and PyLink

Screen: 1024 × 768 pixels, 46.5° × 27° visual angle, 60 cm viewing distance
Face stimuli: 16.35° × 10.22° visual angle
"""

import pylink
import os
import csv
import random
import time
from psychopy import visual, core, event, gui, monitors
from psychopy.constants import FINISHED, PLAYING, STOPPED
import numpy as np

# ==============================================================================
# EXPERIMENT PARAMETERS
# ==============================================================================

# Debug mode - set to True to run without EyeLink connected
DEBUG_MODE = True  # Set to False when running with actual EyeLink

# Display parameters
SCREEN_WIDTH = 1024  # Monitor resolution
SCREEN_HEIGHT = 768  # Monitor resolution
MONITOR_DISTANCE = 60  # cm - distance from participant to screen
SCREEN_WIDTH_CM = 40.6  # Physical screen width in cm
SCREEN_HEIGHT_CM = 30.4  # Physical screen height in cm

# Timing parameters (in seconds)
# --- Trial structure (face familiarity x duration reproduction) ---
PRE_FACE_FIX_DURATION = 0.500  # gray fixation cross BEFORE face onset
FACE_DURATIONS = (0.800, 1.600)  # face stays on screen for one of these
POST_FACE_FIX_DURATION = 0.500  # gray fixation cross AFTER face offset, before reproduction
ITI_DURATION = 0.500  # blank inter-trial interval after J press
REPRODUCTION_KEY = 'j'  # key participant presses to end reproduction phase

# Fixation cross visual params
FIX_SIZE_PIX = 30
FIX_LINE_WIDTH_PIX = 4
FIX_COLOR_GRAY = (0.3, 0.3, 0.3)  # light gray (PsychoPy range -1..1)
FIX_COLOR_BLACK = (-1, -1, -1)  # black

# Validation parameters
FIXATION_TOLERANCE = 1.0  # degrees of visual angle

# Practice parameters
PRACTICE_MIN_CORRECT = 3
PRACTICE_TOTAL_TRIALS = 5
MAX_PRACTICE_SESSIONS = 2

# Main experiment parameters
N_EXPERIMENTAL_TRIALS = 32

# Stimulus parameters
# Interest area size: 16.35° × 10.22° visual angle
FACE_WIDTH_DEG = 16.35  # degrees of visual angle
FACE_HEIGHT_DEG = 10.22  # degrees of visual angle
# These will be converted to pixels based on monitor setup
DOT_RADIUS = 10  # pixels

# ==============================================================================
# FUNCTION TO CHECK FOR EXIT KEYS
# ==============================================================================
def check_for_exit():
    """
    Check whether Shift+E was pressed. If so, close everything and quit
    the program immediately. Call this once per frame in any loop that
    polls the keyboard.
    """
    keys = event.getKeys(keyList=['e'], modifiers=True)
    for key, mods in keys:
        if mods.get('shift'):
            print("Shift+E pressed - exiting program.")
            core.quit()


def wait_for_keys_or_exit(keyList):
    """
    Block until one of the keys in keyList is pressed, just like
    event.waitKeys(keyList=keyList), but still allow Shift+E to quit the
    program at any time while waiting.
    """
    while True:
        check_for_exit()
        keys = event.getKeys(keyList=keyList)
        if keys:
            return keys

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

class DummyEyeLink:
    """Dummy EyeLink class for testing without tracker"""

    def __init__(self):
        print("Running in DEBUG mode - no EyeLink connection")

    def openDataFile(self, fname):
        print(f"[DEBUG] Would open EDF file: {fname}")

    def sendCommand(self, cmd):
        pass

    def sendMessage(self, msg):
        print(f"[DEBUG] Message: {msg}")

    def setOfflineMode(self):
        pass

    def isConnected(self):
        return True

    def getTrackerVersionString(self):
        return "EyeLink 1000 5.0"

    def doTrackerSetup(self):
        print("[DEBUG] Calibration would run here")

    def exitCalibration(self):
        pass

    def startRecording(self, *args):
        return 0

    def stopRecording(self):
        pass

    def getNewestSample(self):
        return None

    def eyeAvailable(self):
        return -1

    def setOfflineMode(self):
        pass

    def closeDataFile(self):
        print("[DEBUG] Closing EDF file")

    def receiveDataFile(self, src, dest):
        print(f"[DEBUG] Would transfer {src} to {dest}")

    def close(self):
        print("[DEBUG] Closing tracker connection")


def pixels_to_degrees(pixels, monitor_distance, screen_width_cm, screen_width_pixels):
    """Convert pixels to degrees of visual angle"""
    pixels_per_cm = screen_width_pixels / screen_width_cm
    size_cm = pixels / pixels_per_cm
    degrees = 2 * np.degrees(np.arctan(size_cm / (2 * monitor_distance)))
    return degrees


def degrees_to_pixels(degrees, monitor_distance, screen_width_cm, screen_width_pixels):
    """Convert degrees of visual angle to pixels"""
    size_cm = 2 * monitor_distance * np.tan(np.radians(degrees / 2))
    pixels_per_cm = screen_width_pixels / screen_width_cm
    pixels = size_cm * pixels_per_cm
    return pixels


def calculate_face_size():
    """Calculate face size in pixels based on visual angle specifications"""
    face_width_pixels = degrees_to_pixels(
        FACE_WIDTH_DEG,
        MONITOR_DISTANCE,
        SCREEN_WIDTH_CM,
        SCREEN_WIDTH
    )
    face_height_pixels = degrees_to_pixels(
        FACE_HEIGHT_DEG,
        MONITOR_DISTANCE,
        SCREEN_WIDTH_CM,  # Use width for consistency in pixels_per_cm calculation
        SCREEN_WIDTH
    )
    return (int(face_width_pixels), int(face_height_pixels))


def verify_display_parameters():
    """Verify that display parameters match specifications and print info"""
    # Calculate screen dimensions in degrees
    screen_width_deg = 2 * np.degrees(np.arctan(SCREEN_WIDTH_CM / (2 * MONITOR_DISTANCE)))
    screen_height_deg = 2 * np.degrees(np.arctan(SCREEN_HEIGHT_CM / (2 * MONITOR_DISTANCE)))

    # Calculate face size in pixels
    face_width_pix, face_height_pix = calculate_face_size()

    print("\n" + "=" * 60)
    print("DISPLAY CONFIGURATION")
    print("=" * 60)
    print(f"Screen resolution: {SCREEN_WIDTH} × {SCREEN_HEIGHT} pixels")
    print(f"Physical screen size: {SCREEN_WIDTH_CM:.1f} × {SCREEN_HEIGHT_CM:.1f} cm")
    print(f"Viewing distance: {MONITOR_DISTANCE} cm")
    print(f"Screen size in visual angle: {screen_width_deg:.1f}° × {screen_height_deg:.1f}°")
    print(f"  (Target: 46.5° × 27°)")
    print(f"\nFace/Interest Area dimensions:")
    print(f"  Visual angle: {FACE_WIDTH_DEG}° × {FACE_HEIGHT_DEG}°")
    print(f"  Pixels: {face_width_pix} × {face_height_pix} pixels")
    print("=" * 60 + "\n")


def setup_eyelink(win, edf_fname):
    """Initialize and configure EyeLink tracker"""

    if DEBUG_MODE:
        print("\n" + "=" * 50)
        print("RUNNING IN DEBUG MODE - NO EYELINK CONNECTED")
        print("=" * 50 + "\n")
        return DummyEyeLink()

    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError:
        print("Could not connect to tracker. Make sure it's on and connected.")
        core.quit()

    # Open EDF file on Host PC
    try:
        el_tracker.openDataFile(edf_fname)
    except RuntimeError as err:
        print(f'Error opening EDF file: {err}')
        el_tracker.close()
        core.quit()

    # Set file preamble
    preamble = f"RECORDED BY PsychoPy\nDot Detection Task"
    el_tracker.sendCommand(f"add_file_preamble_text '{preamble}'")

    # Configure tracker
    el_tracker.setOfflineMode()

    # Get tracker version and set sampling rate
    eyelink_ver = 0
    if el_tracker.isConnected():
        vstr = el_tracker.getTrackerVersionString()
        eyelink_ver = int(vstr.split()[-1].split('.')[0])
        print(f'Running experiment on EyeLink {eyelink_ver}')

    # Set sampling rate (1000 Hz for EyeLink 1000)
    el_tracker.sendCommand("sample_rate 1000")

    # Set link and file event/sample data
    file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
    link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,INPUT'

    if eyelink_ver >= 4:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
    else:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'

    el_tracker.sendCommand(f"file_event_filter = {file_event_flags}")
    el_tracker.sendCommand(f"file_sample_data = {file_sample_flags}")
    el_tracker.sendCommand(f"link_event_filter = {link_event_flags}")
    el_tracker.sendCommand(f"link_sample_data = {link_sample_flags}")

    # Set screen coordinates
    scn_width, scn_height = win.size
    el_coords = f"screen_pixel_coords = 0 0 {scn_width - 1} {scn_height - 1}"
    el_tracker.sendCommand(el_coords)
    el_tracker.sendMessage(f"DISPLAY_COORDS 0 0 {scn_width - 1} {scn_height - 1}")

    return el_tracker


def run_calibration(el_tracker, win):
    """Run calibration and validation"""

    if DEBUG_MODE:
        print("[DEBUG] Skipping calibration in debug mode")
        core.wait(1.0)  # Brief pause to simulate calibration
        return el_tracker

    # Set up calibration graphics
    genv = pylink.EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
    pylink.openGraphicsEx(genv)

    # Calibration type: 9-point
    el_tracker.sendCommand("calibration_type = HV9")

    # Run calibration
    try:
        el_tracker.doTrackerSetup()
    except RuntimeError as err:
        print(f'Calibration error: {err}')
        el_tracker.exitCalibration()

    return el_tracker


def drift_correct(el_tracker, win, position=(0, 0), tolerance_deg=1.0):
    """
    Perform drift correction/validation at specified position
    Returns True if validation passed, False otherwise
    """

    if DEBUG_MODE:
        # In debug mode, just show fixation briefly and pass
        fixation = visual.Circle(win, radius=10, fillColor='white', lineColor='white', pos=position)
        fixation.draw()
        win.flip()
        core.wait(0.5)
        return True

    # Convert tolerance from degrees to pixels
    scn_width, scn_height = win.size
    # Use actual screen dimensions
    tolerance_pix = degrees_to_pixels(tolerance_deg, MONITOR_DISTANCE, SCREEN_WIDTH_CM, scn_width)

    # Draw fixation point
    fixation = visual.Circle(win, radius=10, fillColor='white', lineColor='white', pos=position)
    fixation.draw()
    win.flip()

    # Perform drift correction
    try:
        # Get current gaze position
        el_tracker.sendMessage("DRIFT_CHECK")

        # Start recording to get gaze sample
        error = el_tracker.startRecording(1, 1, 1, 1)
        if error:
            return False

        core.wait(0.1)
        # Get gaze position
        sample = el_tracker.getNewestSample()

        if sample is not None:
            if el_tracker.eyeAvailable() == 1:  # Right eye
                gaze_x = sample.getRightEye().getGaze()[0]
                gaze_y = sample.getRightEye().getGaze()[1]
            elif el_tracker.eyeAvailable() == 0:  # Left eye
                gaze_x = sample.getLeftEye().getGaze()[0]
                gaze_y = sample.getLeftEye().getGaze()[1]
            else:
                el_tracker.stopRecording()
                return False

            # Convert screen center to tracker coordinates
            center_x = scn_width / 2
            center_y = scn_height / 2

            # Calculate deviation
            deviation = np.sqrt((gaze_x - center_x) ** 2 + (gaze_y - center_y) ** 2)

            el_tracker.stopRecording()

            # Check if within tolerance
            if deviation <= tolerance_pix:
                return True
            else:
                # Play error beep
                from psychopy import sound
                error_beep = sound.Sound(800, secs=0.2)
                error_beep.play()
                core.wait(0.2)
                return False
        else:
            el_tracker.stopRecording()
            return False

    except Exception as e:
        print(f"Drift correction error: {e}")
        el_tracker.stopRecording()
        return False


def run_trial_OG(el_tracker, win, trial_num, face_images, is_practice=False):
    """
    Run a single trial

    Parameters:
    - el_tracker: EyeLink tracker object
    - win: PsychoPy window
    - trial_num: trial number
    - face_images: list of 4 face image paths or stimuli
    - is_practice: whether this is a practice trial

    Returns:
    - success: whether trial was completed successfully
    """

    # Calculate face size in pixels
    face_size = calculate_face_size()

    # Drift correction loop
    drift_passed = False
    calibration_count = 0
    max_calibrations = 3

    while not drift_passed and calibration_count < max_calibrations:
        drift_passed = drift_correct(el_tracker, win, tolerance_deg=FIXATION_TOLERANCE)

        if not drift_passed:
            calibration_count += 1
            if calibration_count < max_calibrations:
                # Re-run calibration
                run_calibration(el_tracker, win)
            else:
                print(f"Failed drift correction after {max_calibrations} attempts")
                return False

    # Create face stimuli in 2x2 grid
    DOT_JITTER_RANGE = 50  # pixels
    positions = [(-200, 200), (200, 200), (-200, -200), (200, -200)]  # Top-left, top-right, bottom-left, bottom-right

    face_stims = []
    for i, (face_img, pos) in enumerate(zip(face_images, positions)):
        if isinstance(face_img, str):
            stim = visual.ImageStim(win, image=face_img, pos=pos, size=face_size)
        else:
            stim = face_img
            stim.pos = pos
            stim.size = face_size
        face_stims.append(stim)

    # Create dot stimulus (initially invisible)
    base_position = random.choice(positions)  # Choose which face (e.g., top-left)
    jitter_x = random.uniform(-DOT_JITTER_RANGE, DOT_JITTER_RANGE)  # Random offset in X
    jitter_y = random.uniform(-DOT_JITTER_RANGE, DOT_JITTER_RANGE)  # Random offset in Y
    dot_position = (base_position[0] + jitter_x, base_position[1] + jitter_y)
    dot = visual.Circle(win, radius=DOT_RADIUS, fillColor='grey', lineColor='black',
                        pos=dot_position, opacity=0)

    # Determine random dot onset time
    dot_onset_time = random.uniform(DOT_ONSET_MIN, DOT_ONSET_MAX)

    # Start recording
    el_tracker.sendMessage(f"TRIAL_START {trial_num}")
    if is_practice:
        el_tracker.sendMessage("PRACTICE_TRIAL")

    error = el_tracker.startRecording(1, 1, 1, 1)
    if error:
        print("Recording error")
        return False

    core.wait(0.1)

    # Show faces
    trial_clock = core.Clock()
    trial_clock.reset()

    dot_shown = False
    response_made = False
    response_time = None

    while trial_clock.getTime() < FACE_DISPLAY_DURATION:
        check_for_exit()

        # Draw all faces
        for face_stim in face_stims:
            face_stim.draw()

        # Show dot if it's time
        if trial_clock.getTime() >= dot_onset_time and not dot_shown:
            dot.opacity = 1
            dot_shown = True
            el_tracker.sendMessage(f"DOT_ONSET {dot_onset_time:.3f} POS_X_{dot_position[0]}_Y_{dot_position[1]}")

        if dot_shown:
            dot.draw()

        win.flip()

        # Check for response
        if not response_made:
            keys = event.getKeys(keyList=['left', 'right'], timeStamped=trial_clock)
            if keys:
                response_made = True
                response_time = keys[0][1]
                el_tracker.sendMessage(f"RESPONSE {response_time:.3f}")

    # Blank fixation period
    fixation = visual.Circle(win, radius=10, fillColor='white', lineColor='white')
    fixation.draw()
    win.flip()

    el_tracker.sendMessage("BLANK_FIXATION_START")
    core.wait(BLANK_FIXATION_DURATION)
    el_tracker.sendMessage("BLANK_FIXATION_END")

    # Stop recording
    el_tracker.stopRecording()
    el_tracker.sendMessage(f"TRIAL_END {trial_num}")

    # Calculate if response was correct (within reasonable time after dot)
    correct = False
    if response_made and response_time is not None:
        rt = response_time - dot_onset_time
        if 0.1 < rt < 2.0:  # Response between 100ms and 2000ms after dot
            correct = True

    return correct


def _make_fix_cross(win, color):
    """Build a fixation cross (two crossed lines) at screen center."""
    horiz = visual.Line(win,
                        start=(-FIX_SIZE_PIX / 2, 0),
                        end=(FIX_SIZE_PIX / 2, 0),
                        lineColor=color,
                        lineWidth=FIX_LINE_WIDTH_PIX)
    vert = visual.Line(win,
                       start=(0, -FIX_SIZE_PIX / 2),
                       end=(0, FIX_SIZE_PIX / 2),
                       lineColor=color,
                       lineWidth=FIX_LINE_WIDTH_PIX)
    return [horiz, vert]


def _draw_fix(fix_components):
    for c in fix_components:
        c.draw()


def _classify_face(face_path):
    """
    Return (familiarity, source) for a face image path.

    Assumes folder layout:
        experimental_faces/uk_celebrities/...
        experimental_faces/israeli_celebrities/...
        experimental_faces/database_faces/...

    UK + Israeli  -> familiar
    database      -> unfamiliar
    """
    parent = os.path.basename(os.path.dirname(face_path)).lower()
    if 'uk' in parent:
        return 'familiar', 'uk'
    elif 'israeli' in parent:
        return 'familiar', 'israeli'
    elif 'database' in parent:
        return 'unfamiliar', 'database'
    else:
        return 'unknown', parent


def run_trial(el_tracker, win, trial_num, face_image, face_duration_s, is_practice=False):
    """
    Run a single trial of the face familiarity x duration reproduction task.

    Sequence
    --------
      (a) light-gray fixation cross  -- PRE_FACE_FIX_DURATION (500 ms)
      (b) single face                -- face_duration_s (800 or 1600 ms)
      (c) light-gray fixation cross  -- POST_FACE_FIX_DURATION (500 ms)
      (d) BLACK fixation cross       -- stays on screen until participant
                                        presses REPRODUCTION_KEY ('j');
                                        the elapsed time = reproduced duration
      (e) blank ITI                  -- ITI_DURATION (500 ms)

    Parameters
    ----------
    el_tracker      : EyeLink tracker (real or DummyEyeLink)
    win             : PsychoPy window
    trial_num       : int or str (used as trial identifier)
    face_image      : str, path to the face image for this trial
    face_duration_s : float, stimulus duration in seconds (0.8 or 1.6)
    is_practice    : bool

    Returns
    -------
    trial_data : dict or None on abort
    """

    # ----- Drift correction -----
    drift_passed = False
    calibration_count = 0
    max_calibrations = 3
    while not drift_passed and calibration_count < max_calibrations:
        drift_passed = drift_correct(el_tracker, win, tolerance_deg=FIXATION_TOLERANCE)
        if not drift_passed:
            calibration_count += 1
            if calibration_count < max_calibrations:
                run_calibration(el_tracker, win)
            else:
                print(f"Failed drift correction after {max_calibrations} attempts")
                return None

    # ----- Build stimuli -----
    face_size = calculate_face_size()
    face_stim = visual.ImageStim(win, image=face_image, pos=(0, 0), size=face_size)

    fix_gray = _make_fix_cross(win, FIX_COLOR_GRAY)
    fix_black = _make_fix_cross(win, FIX_COLOR_BLACK)

    familiarity, face_source = _classify_face(face_image)

    # ----- Start recording -----
    el_tracker.sendMessage(f"TRIAL_START {trial_num}")
    if is_practice:
        el_tracker.sendMessage("PRACTICE_TRIAL")
    el_tracker.sendMessage(f"FACE_IMAGE {os.path.basename(face_image)}")
    el_tracker.sendMessage(f"FACE_FAMILIARITY {familiarity}")
    el_tracker.sendMessage(f"FACE_SOURCE {face_source}")
    el_tracker.sendMessage(f"FACE_DURATION_PLANNED_MS {int(face_duration_s * 1000)}")

    error = el_tracker.startRecording(1, 1, 1, 1)
    if error:
        print("Recording error")
        return None
    core.wait(0.1)

    # ===== (a) Pre-face gray fixation =====
    clk = core.Clock()
    clk.reset()
    el_tracker.sendMessage("PRE_FACE_FIX_ONSET")
    while clk.getTime() < PRE_FACE_FIX_DURATION:
        check_for_exit()
        _draw_fix(fix_gray)
        win.flip()
        if event.getKeys(keyList=['escape']):
            el_tracker.stopRecording()
            return None

    # ===== (b) Face =====
    face_clk = core.Clock()
    face_clk.reset()
    el_tracker.sendMessage("FACE_ONSET")
    while face_clk.getTime() < face_duration_s:
        check_for_exit()
        face_stim.draw()
        win.flip()
        if event.getKeys(keyList=['escape']):
            el_tracker.stopRecording()
            return None
    actual_face_duration = face_clk.getTime()
    el_tracker.sendMessage("FACE_OFFSET")

    # ===== (c) Post-face gray fixation =====
    post_clk = core.Clock()
    post_clk.reset()
    el_tracker.sendMessage("POST_FACE_FIX_ONSET")
    while post_clk.getTime() < POST_FACE_FIX_DURATION:
        check_for_exit()
        _draw_fix(fix_gray)
        win.flip()
        if event.getKeys(keyList=['escape']):
            el_tracker.stopRecording()
            return None

    # ===== (d) Black fixation cross — reproduction phase =====
    # Stays on screen until participant presses REPRODUCTION_KEY ('j').
    event.clearEvents()
    repro_clk = core.Clock()
    repro_clk.reset()
    el_tracker.sendMessage("REPRODUCTION_ONSET")

    reproduced_duration = None
    while reproduced_duration is None:
        check_for_exit()
        _draw_fix(fix_black)
        win.flip()
        keys = event.getKeys(keyList=[REPRODUCTION_KEY, 'escape'], timeStamped=repro_clk)
        if keys:
            k_name, k_time = keys[0]
            if k_name == 'escape':
                el_tracker.stopRecording()
                return None
            if k_name == REPRODUCTION_KEY:
                reproduced_duration = k_time
                break

    el_tracker.sendMessage(f"REPRODUCTION_RESPONSE {reproduced_duration:.4f}")

    # ===== (e) Blank ITI =====
    iti_clk = core.Clock()
    iti_clk.reset()
    el_tracker.sendMessage("ITI_ONSET")
    while iti_clk.getTime() < ITI_DURATION:
        check_for_exit()
        win.flip()  # blank screen (window background)
    el_tracker.sendMessage("ITI_OFFSET")

    # ----- Stop recording -----
    el_tracker.stopRecording()
    el_tracker.sendMessage(f"TRIAL_END {trial_num}")

    # ----- Trial data -----
    trial_data = {
        'trial_num': trial_num,
        'is_practice': is_practice,
        'face_image': face_image,
        'face_familiarity': familiarity,  # 'familiar' | 'unfamiliar'
        'face_source': face_source,  # 'uk' | 'israeli' | 'database'
        'face_duration_planned_s': face_duration_s,
        'face_duration_actual_s': actual_face_duration,
        'reproduced_duration_s': reproduced_duration,
        'reproduction_error_s': reproduced_duration - face_duration_s,
    }
    return trial_data


def run_practice(el_tracker, win, practice_faces):
    """
    Run practice session(s) for the face familiarity x duration reproduction task.
    Returns True if participant completes a practice session.
    """
    session_count = 0

    while session_count < MAX_PRACTICE_SESSIONS:
        # Instructions
        instructions = visual.TextStim(win,
                                       text=f"Practice Session {session_count + 1}\n\n"
                                            "On each trial a face will appear briefly in the center.\n"
                                            "When the fixation cross turns BLACK, press the 'J' key\n"
                                            "when you feel the same amount of time has elapsed as\n"
                                            "the face was on the screen.\n\n"
                                            "Press SPACE to begin.",
                                       height=30, wrapWidth=1000)
        instructions.draw()
        win.flip()
        wait_for_keys_or_exit(['space'])

        # Build a balanced practice list: every face x every duration, shuffled,
        # then cap at PRACTICE_TOTAL_TRIALS.
        practice_trials = [(face, dur) for face in practice_faces for dur in FACE_DURATIONS]
        random.shuffle(practice_trials)
        practice_trials = practice_trials[:PRACTICE_TOTAL_TRIALS]

        completed_count = 0
        for trial_idx, (face_img, dur) in enumerate(practice_trials):
            trial_data = run_trial(
                el_tracker, win,
                trial_num=f"practice_{session_count}_{trial_idx}",
                face_image=face_img,
                face_duration_s=dur,
                is_practice=True,
            )
            if trial_data is not None:
                completed_count += 1

        # In a duration-reproduction task there's no "correct/incorrect" gating,
        # so we just require the participant to have completed the trials.
        if completed_count >= PRACTICE_MIN_CORRECT:
            feedback = visual.TextStim(win,
                                       text=f"Practice complete!\n\n"
                                            f"You completed {completed_count} of {PRACTICE_TOTAL_TRIALS} practice trials.\n\n"
                                            "Press SPACE to continue to the main experiment.",
                                       height=30, wrapWidth=1000)
            feedback.draw()
            win.flip()
            wait_for_keys_or_exit(['space'])
            return True
        else:
            session_count += 1
            if session_count < MAX_PRACTICE_SESSIONS:
                feedback = visual.TextStim(win,
                                           text=f"You completed {completed_count} of {PRACTICE_TOTAL_TRIALS} practice trials.\n\n"
                                                "Let's try another practice session.\n\n"
                                                "Press SPACE to continue.",
                                           height=30, wrapWidth=1000)
                feedback.draw()
                win.flip()
                wait_for_keys_or_exit(['space'])

    # End-of-practice message
    feedback = visual.TextStim(win,
                               text="Practice session complete.\n\n"
                                    "Press SPACE to continue to the main experiment.",
                               height=30, wrapWidth=1000)
    feedback.draw()
    win.flip()
    wait_for_keys_or_exit(['space'])
    return False


# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

def main():
    # Verify and display configuration
    verify_display_parameters()

    # Get participant info
    exp_info = {'Participant ID': '', 'Session': '001'}
    dlg = gui.DlgFromDict(dictionary=exp_info, title='Face Detection Experiment')
    if not dlg.OK:
        core.quit()

    # Create data folder if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # EDF filename (must be <= 8 characters + .edf)
    edf_fname = f"{exp_info['Participant ID'][:4]}{exp_info['Session']}.edf"

    # Create behavioral data filename
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    behavioral_fname = os.path.join('data',
                                    f"{exp_info['Participant ID']}_sess{exp_info['Session']}_{timestamp}_behavioral.csv")

    # Set up window
    win = visual.Window(
        size=(SCREEN_WIDTH, SCREEN_HEIGHT),
        fullscr=not DEBUG_MODE,  # Windowed mode in debug
        screen=0,
        allowGUI=False,
        color=[-1, -1, -1],
        units='pix',
    )

    # Hide mouse cursor
    win.mouseVisible = False

    # Initialize EyeLink
    el_tracker = setup_eyelink(win, edf_fname)

    # Run calibration
    run_calibration(el_tracker, win)

    # Load face stimuli
    # YOU NEED TO REPLACE THIS WITH YOUR ACTUAL FACE IMAGE PATHS
    practice_face_folder = "practice_faces"
    experimental_face_folder = "experimental_faces"

    # Example: Load practice faces
    practice_faces = []
    if os.path.exists(practice_face_folder):
        practice_face_files = [os.path.join(practice_face_folder, f)
                               for f in os.listdir(practice_face_folder)
                               if f.endswith(('.jpg', '.png', '.bmp'))]
        practice_faces = practice_face_files
    else:
        print(f"Warning: Practice face folder '{practice_face_folder}' not found!")
        print("Please create the folder and add practice face images.")
        el_tracker.close()
        win.close()
        core.quit()

    # ----- Load experimental faces -----
    # We expect experimental_face_folder to contain three subfolders:
    #   experimental_faces/uk_celebrities/
    #   experimental_faces/israeli_celebrities/
    #   experimental_faces/database_faces/
    # _classify_face() reads familiarity from the parent folder of each image.
    experimental_faces = []
    if os.path.exists(experimental_face_folder):
        for root, dirs, files in os.walk(experimental_face_folder):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    experimental_faces.append(os.path.join(root, f))
    if not experimental_faces:
        print(f"Warning: no experimental faces found under '{experimental_face_folder}'!")
        print("Expected subfolders: uk_celebrities/, israeli_celebrities/, database_faces/")
        el_tracker.close()
        win.close()
        core.quit()

    # Initialize list to store all behavioral data
    all_behavioral_data = []

    # Run practice
    print("Running practice trials...")
    practice_data = run_practice(el_tracker, win, practice_faces)

    # Main experiment instructions
    instructions = visual.TextStim(win,
                                   text="Main Experiment\n\n"
                                        "A face will appear briefly on each trial.\n"
                                        "When the fixation cross turns BLACK, press 'J' when you feel\n"
                                        "the same amount of time has elapsed as the face was shown.\n\n"
                                        "Press SPACE to begin.",
                                   height=30, wrapWidth=1000)
    instructions.draw()
    win.flip()
    wait_for_keys_or_exit(['space'])

    # ----- Build the experimental trial list -----
    # Cross every face with both durations, then shuffle.  This keeps the
    # 800 ms and 1600 ms conditions exactly balanced and randomizes the
    # order across the experiment, as requested.
    trial_face_list = [(face, dur)
                       for face in experimental_faces
                       for dur in FACE_DURATIONS]
    random.shuffle(trial_face_list)
    # Cap to N_EXPERIMENTAL_TRIALS
    trial_face_list = trial_face_list[:N_EXPERIMENTAL_TRIALS]

    # ----- Run experimental trials -----
    for trial_num, (face_img, dur) in enumerate(trial_face_list):
        trial_data = run_trial(
            el_tracker, win,
            trial_num=trial_num,
            face_image=face_img,
            face_duration_s=dur,
            is_practice=False,
        )

        if trial_data is not None:
            # Add participant info to each trial
            trial_data['participant_id'] = exp_info['Participant ID']
            trial_data['session'] = exp_info['Session']
            all_behavioral_data.append(trial_data)

        # Allow Shift+E to fully exit, or escape to stop the experiment
        check_for_exit()
        if 'escape' in event.getKeys():
            break

    # End experiment
    el_tracker.sendMessage("EXPERIMENT_END")

    # Close EDF file and transfer
    el_tracker.setOfflineMode()
    core.wait(0.1)
    el_tracker.closeDataFile()

    # Save behavioral data to CSV
    if all_behavioral_data:
        fieldnames = ['participant_id', 'session', 'trial_num', 'is_practice',
                      'face_image', 'face_familiarity', 'face_source',
                      'face_duration_planned_s', 'face_duration_actual_s',
                      'reproduced_duration_s', 'reproduction_error_s']

        with open(behavioral_fname, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_behavioral_data)

        print(f"Behavioral data saved to: {behavioral_fname}")

    # Transfer EDF file
    local_edf = os.path.join('data', f"{exp_info['Participant ID']}_{exp_info['Session']}.edf")
    try:
        el_tracker.receiveDataFile(edf_fname, local_edf)
        print(f"EDF file saved to: {local_edf}")
    except RuntimeError as error:
        print(f"Error transferring EDF file: {error}")

    # Close connection
    el_tracker.close()

    # Thank you message
    thanks = visual.TextStim(win, text="Thank you for participating!\n\nPress any key to exit.",
                             height=30)
    thanks.draw()
    win.flip()
    event.waitKeys()

    # Clean up
    win.close()
    core.quit()


if __name__ == "__main__":
    main()