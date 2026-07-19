"""
EyeLink 1000 Face Detection Experiment
Using PsychoPy and PyLink

Screen: 1920 x 1080 pixels (Propixx), 60 cm viewing distance
Face stimuli: 16.35° × 10.22° visual angle

V8 changes vs V7:
  - run_calibration(): fixed genv creation to temporarily chdir into
    CalibrationSounds so EyeLinkCoreGraphicsPsychoPy's internal default
    sound objects (_target_beep etc.) initialize correctly, then explicitly
    call setCalibrationSounds() with full paths.
  - setup_eyelink(): now prints the actual connection error instead of
    swallowing it.
  - Added a background emergency-kill watchdog (Ctrl+Alt+Q) that works even
    while blocked inside pylink calls like doTrackerSetup(), where the
    normal Shift+Q check_for_exit() cannot run because control has passed
    to SR Research's own event loop.
"""

import pylink
import os
import csv
import random
import time
import threading
from psychopy import visual, core, event, gui, monitors
from psychopy.constants import FINISHED, PLAYING, STOPPED
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy

import EyeLinkCoreGraphicsPsychoPy as _elcg_module
print(f"[DEBUG] Loading EyeLinkCoreGraphicsPsychoPy from: {_elcg_module.__file__}")

import numpy as np

# ==============================================================================
# PROPIXX PROJECTOR SETUP
# ==============================================================================
USE_PROPIXX = True   # set to False to run on a normal monitor for testing

try:
    import propixx_flicker as fl
    _HAVE_PROPIXX_MODULE = True
except ImportError:
    _HAVE_PROPIXX_MODULE = False
    print("Warning: propixx_flicker module not found. Will fall back to "
          "SCREEN_WIDTH/HEIGHT constants for window size.")


def init_propixx_normal_mode():
    """
    Put the Propixx into normal 120 Hz RGB mode (NOT 1440 Hz quadrant mode).
    Returns a 'restore' callable that should be called at exit.
    """
    if not USE_PROPIXX:
        print("USE_PROPIXX = False -- skipping projector init.")
        return lambda: None

    try:
        from pypixxlib.propixx import PROPixx
        dev = PROPixx()
        dev.setDlpSequencerProgram('RGB')
        dev.updateRegisterCache()
        print("Propixx initialized in normal 120 Hz RGB mode.")

        def restore():
            try:
                dev.setDlpSequencerProgram('RGB')
                dev.updateRegisterCache()
                dev.close()
                print("Propixx restored to default mode.")
            except Exception as e:
                print(f"Propixx restore warning: {e}")

        return restore

    except ImportError:
        print("pypixxlib not installed -- assuming Propixx is already in "
              "normal mode (set manually via VPutil).")
        return lambda: None
    except Exception as e:
        print(f"Propixx init error: {e}")
        print("Assuming Propixx is already in normal mode.")
        return lambda: None


# ==============================================================================
# EMERGENCY KILL WATCHDOG
# ==============================================================================

def _emergency_kill_watchdog():
    """
    Runs in a background thread and force-kills the whole process on
    Ctrl+Alt+Q, no matter what the main thread is doing -- including while
    blocked inside pylink calls like doTrackerSetup(), where normal
    check_for_exit() cannot run because control has passed to SR Research's
    own event loop.

    NOTE: This is a hard kill (os._exit) -- it skips normal cleanup
    (Propixx restore, EyeLink close, window close, in-progress CSV writes).
    Data from already-completed trials that was already written to disk is
    safe; use only when the program is genuinely unresponsive.

    Requires: pip install keyboard
    On Windows, global key hooks generally require running as Administrator.
    """
    try:
        import keyboard
    except ImportError:
        print("[WARNING] 'keyboard' module not installed -- emergency kill "
              "hotkey (Ctrl+Alt+Q) will NOT work. Install with: "
              "pip install keyboard")
        return

    def _kill():
        print("\n[EMERGENCY KILL] Ctrl+Alt+Q pressed - force-terminating process.")
        os._exit(1)

    try:
        keyboard.add_hotkey('ctrl+alt+q', _kill)
        print("[DEBUG] Emergency kill hotkey armed: Ctrl+Alt+Q")
        keyboard.wait()  # blocks this thread only, listens forever
    except Exception as e:
        print(f"[WARNING] Emergency kill hotkey failed to arm: {e}")


# ==============================================================================
# EXPERIMENT PARAMETERS
# ==============================================================================

# Debug mode - set to True to run without EyeLink connected
DEBUG_MODE = False
DEBUG_MODE_SKIP = True

# Windowed mode - independent of DEBUG_MODE.
# Set WINDOWED_MODE = True to run in a smaller non-fullscreen window
# (useful for piloting and demos; EyeLink still connects and records).
# DEBUG_MODE always forces windowed regardless of this flag.
WINDOWED_MODE = False
WINDOWED_SIZE = (1280, 720)  # used only when WINDOWED_MODE = True or DEBUG_MODE = True

# Display parameters (used for face size calculations and EyeLink coord mapping)
# Set these to the ACTUAL Propixx projected image dimensions at your viewing distance.
SCREEN_WIDTH = 1920   # pixels (Propixx native)
SCREEN_HEIGHT = 1080  # pixels (Propixx native)
MONITOR_DISTANCE = 60   # cm
SCREEN_WIDTH_CM = 40.6  # physical width of projected image in cm  <-- update to actual
SCREEN_HEIGHT_CM = 22.8 # physical height of projected image in cm <-- update to actual

# Timing parameters (seconds)
PRE_FACE_FIX_DURATION  = 0.500  # gray fixation cross before face
FACE_DURATIONS         = (0.800, 1.600)  # face durations
POST_FACE_FIX_DURATION = 0.500  # gray fixation cross after face, before reproduction
ITI_DURATION           = 0.500  # blank ITI after SPACE press
REPRODUCTION_KEY       = 'space'

# Fixation cross visual params
FIX_SIZE_PIX            = 30
FIX_LINE_WIDTH_PIX      = 4
FIX_COLOR_GRAY          = (0.3, 0.3, 0.3)
FIX_COLOR_REPRODUCTION  = (1, 1, 1)      # white — visible on black background
FIX_SIZE_REPRODUCTION_PIX = 40

# Practice demo cross (shown in tutorial screens only)
PRACTICE_DEMO_LARGE_CROSS_SIZE_PIX = FIX_SIZE_REPRODUCTION_PIX
PRACTICE_DEMO_LARGE_CROSS_COLOR    = (1, 1, 1)

# Tutorial font
_PSYCHOPY_FONT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(visual.__file__)), 'assets', 'fonts')
TUTORIAL_FONT_NAME = 'Noto Sans'
_candidate_font_files = [
    os.path.join(_PSYCHOPY_FONT_DIR, 'NotoSans-Regular.ttf'),
    os.path.join(_PSYCHOPY_FONT_DIR, 'NotoSans-Bold.ttf'),
    os.path.join(_PSYCHOPY_FONT_DIR, 'NotoSans-Italic.ttf'),
    os.path.join(_PSYCHOPY_FONT_DIR, 'NotoSans-BoldItalic.ttf'),
]
TUTORIAL_FONT_FILES = [f for f in _candidate_font_files if os.path.isfile(f)]

# ---------------------------------------------------------------------------
# Compatibility shim: TextStim keyword args changed between PsychoPy versions
# ---------------------------------------------------------------------------
import inspect as _inspect

_TextStim_orig_init = visual.TextStim.__init__
_textstim_params = set(_inspect.signature(_TextStim_orig_init).parameters.keys())


def _textstim_compat_init(self, *args, **kwargs):
    if 'alignText' not in _textstim_params:
        align_val = kwargs.pop('alignText', None)
        if align_val is not None and 'alignHoriz' not in kwargs:
            kwargs['alignHoriz'] = align_val
    if 'anchorHoriz' not in _textstim_params:
        kwargs.pop('anchorHoriz', None)
    if 'anchorVert' not in _textstim_params:
        kwargs.pop('anchorVert', None)
    _TextStim_orig_init(self, *args, **kwargs)


visual.TextStim.__init__ = _textstim_compat_init

# ---------------------------------------------------------------------------
# Compatibility shim: Slider color args changed between PsychoPy versions
# ---------------------------------------------------------------------------
from psychopy.visual import Slider as _Slider_orig

_Slider_orig_init = _Slider_orig.__init__
_slider_params    = set(_inspect.signature(_Slider_orig_init).parameters.keys())
_SLIDER_NEW_COLOR_ARGS = ('fillColor', 'borderColor', 'markerColor', 'labelColor')


def _slider_compat_init(self, *args, **kwargs):
    for arg in _SLIDER_NEW_COLOR_ARGS:
        if arg not in _slider_params:
            val = kwargs.pop(arg, None)
            if arg == 'markerColor' and val is not None and 'color' not in kwargs:
                kwargs['color'] = val
    _Slider_orig_init(self, *args, **kwargs)

_Slider_orig.__init__ = _slider_compat_init

# EyeLink / trial params
FIXATION_TOLERANCE     = 1.0   # degrees for drift correction tolerance
PRACTICE_MIN_CORRECT   = 3
PRACTICE_TOTAL_TRIALS  = 5
MAX_PRACTICE_SESSIONS  = 2
N_EXPERIMENTAL_TRIALS  = 32

# Face stimulus size in visual angle
FACE_WIDTH_DEG  = 16.35
FACE_HEIGHT_DEG = 10.22
DOT_RADIUS      = 10   # pixels (unused in main task, kept for legacy)


# ==============================================================================
# KEY / EXIT HELPERS
# ==============================================================================

def check_for_exit():
    """Shift+Q quits the experiment immediately."""
    keys = event.getKeys(keyList=['q'], modifiers=True)
    if keys:
        print(f"[DEBUG] 'q' detected, raw keys+mods: {keys}")
    for key, mods in keys:
        if mods.get('shift'):
            print("Shift+Q pressed - exiting program.")
            core.quit()
        else:
            print(f"[DEBUG] 'q' seen but shift NOT flagged. mods dict = {mods}")


def check_for_skip():
    """Shift+S skips the current phase (DEBUG_MODE only)."""
    if not DEBUG_MODE_SKIP:
        return False
    keys = event.getKeys(keyList=['s'], modifiers=True)
    for key, mods in keys:
        if mods.get('shift'):
            print("[DEBUG] Shift+S pressed - skipping current phase.")
            return True
    return False


def safe_flip(win):
    """win.flip() with retry on Mac/pyglet Cocoa AttributeError."""
    from psychopy import core as _core, event as _ev
    for _ in range(3):
        try:
            win.flip()
            return
        except AttributeError:
            _ev.clearEvents()
            _core.wait(0.05)
    win.flip()


def wait_for_keys_or_exit(keyList):
    """Block until one of keyList is pressed; 'space' mapped to 'return'."""
    extended = list(set('return' if k == 'space' else k for k in keyList))
    while True:
        check_for_exit()
        keys = event.getKeys(keyList=extended)
        if keys:
            return keys


# ==============================================================================
# DUMMY EYELINK (debug mode)
# ==============================================================================

class DummyEyeLink:
    """Drop-in replacement used when DEBUG_MODE = True."""

    def __init__(self):
        print("Running in DEBUG mode - no EyeLink connection")

    def openDataFile(self, fname):      print(f"[DEBUG] Would open EDF: {fname}")
    def sendCommand(self, cmd):         pass
    def sendMessage(self, msg):         print(f"[DEBUG] EDF msg: {msg}")
    def setOfflineMode(self):           pass
    def isConnected(self):              return True
    def getTrackerVersionString(self):  return "EyeLink 1000 5.0"
    def doTrackerSetup(self):           print("[DEBUG] Calibration skipped")
    def doDriftCorrect(self, *a, **k):  return 0
    def exitCalibration(self):          pass
    def startRecording(self, *args):    return 0
    def stopRecording(self):            pass
    def getNewestSample(self):          return None
    def eyeAvailable(self):             return -1
    def closeDataFile(self):            print("[DEBUG] Closing EDF")
    def receiveDataFile(self, src, dst):print(f"[DEBUG] Would transfer {src} → {dst}")
    def close(self):                    print("[DEBUG] Closing tracker")


# ==============================================================================
# DISPLAY UTILITIES
# ==============================================================================

def pixels_to_degrees(pixels, monitor_distance, screen_width_cm, screen_width_pixels):
    pixels_per_cm = screen_width_pixels / screen_width_cm
    size_cm = pixels / pixels_per_cm
    return 2 * np.degrees(np.arctan(size_cm / (2 * monitor_distance)))


def degrees_to_pixels(degrees, monitor_distance, screen_width_cm, screen_width_pixels):
    size_cm = 2 * monitor_distance * np.tan(np.radians(degrees / 2))
    pixels_per_cm = screen_width_pixels / screen_width_cm
    return size_cm * pixels_per_cm


def calculate_face_size():
    """Return (width_px, height_px) for the face stimulus."""
    w = degrees_to_pixels(FACE_WIDTH_DEG,  MONITOR_DISTANCE, SCREEN_WIDTH_CM,  SCREEN_WIDTH)
    h = degrees_to_pixels(FACE_HEIGHT_DEG, MONITOR_DISTANCE, SCREEN_WIDTH_CM,  SCREEN_WIDTH)
    return (int(w), int(h))


def verify_display_parameters():
    sw_deg = 2 * np.degrees(np.arctan(SCREEN_WIDTH_CM  / (2 * MONITOR_DISTANCE)))
    sh_deg = 2 * np.degrees(np.arctan(SCREEN_HEIGHT_CM / (2 * MONITOR_DISTANCE)))
    fw, fh = calculate_face_size()
    print("\n" + "=" * 60)
    print("DISPLAY CONFIGURATION")
    print("=" * 60)
    print(f"Screen resolution : {SCREEN_WIDTH} × {SCREEN_HEIGHT} px")
    print(f"Physical size     : {SCREEN_WIDTH_CM:.1f} × {SCREEN_HEIGHT_CM:.1f} cm")
    print(f"Viewing distance  : {MONITOR_DISTANCE} cm")
    print(f"Screen visual angle: {sw_deg:.1f}° × {sh_deg:.1f}°")
    print(f"Face visual angle : {FACE_WIDTH_DEG}° × {FACE_HEIGHT_DEG}°")
    print(f"Face pixels       : {fw} × {fh} px")
    print("=" * 60 + "\n")


# ==============================================================================
# EYELINK SETUP
# ==============================================================================

def setup_eyelink(win, edf_fname):
    """Connect to tracker, configure sampling, open EDF file."""
    if DEBUG_MODE:
        print("\n" + "=" * 50)
        print("RUNNING IN DEBUG MODE - NO EYELINK CONNECTED")
        print("=" * 50 + "\n")
        return DummyEyeLink()

    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as err:
        print(f"Could not connect to EyeLink: {err}")
        print("Check Ethernet cable, Display PC IP (should be 100.1.1.2/24), "
              "and that the Host PC is at the main tracker screen.")
        core.quit()

    try:
        el_tracker.openDataFile(edf_fname)
    except RuntimeError as err:
        print(f"Error opening EDF file: {err}")
        el_tracker.close()
        core.quit()

    el_tracker.sendCommand(
        "add_file_preamble_text 'RECORDED BY PsychoPy - Face Familiarity Duration Reproduction'")
    el_tracker.setOfflineMode()

    eyelink_ver = 0
    if el_tracker.isConnected():
        vstr = el_tracker.getTrackerVersionString()
        eyelink_ver = int(vstr.split()[-1].split('.')[0])
        print(f"Running experiment on EyeLink {eyelink_ver}")

    el_tracker.sendCommand("sample_rate 1000")

    file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
    link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,INPUT'
    if eyelink_ver >= 4:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
    else:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'

    el_tracker.sendCommand(f"file_event_filter = {file_event_flags}")
    el_tracker.sendCommand(f"file_sample_data  = {file_sample_flags}")
    el_tracker.sendCommand(f"link_event_filter = {link_event_flags}")
    el_tracker.sendCommand(f"link_sample_data  = {link_sample_flags}")

    scn_w, scn_h = win.size
    el_tracker.sendCommand(f"screen_pixel_coords = 0 0 {scn_w - 1} {scn_h - 1}")
    el_tracker.sendMessage(f"DISPLAY_COORDS 0 0 {scn_w - 1} {scn_h - 1}")

    return el_tracker


# ==============================================================================
# CALIBRATION  (V8: fixed genv/sound init — chdir into CalibrationSounds while
#                    constructing genv, then explicitly setCalibrationSounds())
# ==============================================================================

def run_calibration(el_tracker, win):
    """
    Run EyeLink camera setup / calibration / validation.
    """
    if DEBUG_MODE:
        print("[DEBUG] Skipping calibration in debug mode")
        core.wait(0.5)
        return el_tracker

    # Build path to CalibrationSounds next to this script
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _sounds_dir = os.path.join(_script_dir, 'CalibrationSounds')

    # EyeLinkCoreGraphicsPsychoPy's __init__ creates its default sound
    # objects (_target_beep etc.) by loading 'type.wav' etc. relative to
    # the CURRENT WORKING DIRECTORY. If those files aren't found there,
    # the sound objects silently fail to get created, and any later call
    # to setCalibrationSounds() raises AttributeError. Temporarily chdir
    # into CalibrationSounds so the default load succeeds.
    _prev_cwd = os.getcwd()
    try:
        os.chdir(_sounds_dir)
        genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
    finally:
        os.chdir(_prev_cwd)
    print(f"[DEBUG] genv created: {genv}")

    # openGraphicsEx MUST come before any genv configuration calls —
    # the internal sound/target objects (_target_beep etc.) are only
    # created inside openGraphicsEx, so calling setCalibrationSounds()
    # or setTargetType() before it raises AttributeError.
    pylink.openGraphicsEx(genv)
    print("[DEBUG] openGraphicsEx called")

    # Calibration target appearance
    genv.setCalibrationColors('white', 'black')
    genv.setTargetType('circle')
    genv.setTargetSize(24)

    # Point the library explicitly at the calibration sound files, using
    # full paths so it doesn't matter what the current working directory is.
    target_beep = os.path.join(_sounds_dir, 'type.wav')
    good_beep   = os.path.join(_sounds_dir, 'qbeep.wav')
    error_beep  = os.path.join(_sounds_dir, 'error.wav')
    if all(os.path.isfile(p) for p in (target_beep, good_beep, error_beep)):
        genv.setCalibrationSounds(target_beep, good_beep, error_beep)
        print(f"[DEBUG] Calibration sounds loaded from {_sounds_dir}")
    else:
        print(f"[WARNING] Calibration sound files missing in {_sounds_dir} "
              f"— calibration may crash without them.")

    el_tracker.sendCommand("calibration_type = HV9")

    # Force keyboard focus to the PsychoPy window using the Windows API.
    # win.winHandle.activate() alone is insufficient on Windows when another
    # application (e.g. PyCharm) holds the foreground lock — the EyeLink menu
    # is drawn but keypresses (Enter, C, V, O) go to the wrong window.
    # try:
    #     import ctypes
    #     hwnd = win._hw_handle   # pyglet HWND for this window
    #     ctypes.windll.user32.SetForegroundWindow(hwnd)
    #     ctypes.windll.user32.BringWindowToTop(hwnd)
    #     ctypes.windll.user32.SetFocus(hwnd)
    #     print("[DEBUG] ctypes focus applied to PQsychoPy window")
    # except Exception as e:
    #     print(f"[WARNING] ctypes focus failed: {e}")
    #     try:
    #         win.winHandle.activate()
    #         win.winHandle.set_visible(True)
    #     except Exception:
    #         pass

    try:
        print("[DEBUG] calling doTrackerSetup...", flush=True)
        el_tracker.doTrackerSetup()
        print("[DEBUG] doTrackerSetup returned", flush=True)
    except Exception as err:
        import traceback
        traceback.print_exc()
        el_tracker.exitCalibration()

    return el_tracker


# ==============================================================================
# DRIFT CORRECTION  (V7: replaced broken manual implementation with standard
#                        pylink.doDriftCorrect() call)
# ==============================================================================

def drift_correct(el_tracker, win, position=(0, 0), tolerance_deg=1.0):
    """
    Perform drift correction using pylink.doDriftCorrect().

    This is the standard SR Research approach. The tracker draws the target
    on the Host PC display, waits for the participant to fixate, applies the
    correction, and returns a result code.

    Returns True on success, False if the operator pressed Esc to go to setup
    (the caller then triggers recalibration).
    """
    if DEBUG_MODE:
        fixation = visual.Circle(win, radius=10, fillColor='white',
                                 lineColor='white', pos=position)
        fixation.draw()
        win.flip()
        core.wait(0.5)
        return True

    # Convert PsychoPy center-origin coords to EyeLink top-left-origin coords.
    scn_w, scn_h = win.size
    el_x = int(scn_w / 2 + position[0])
    el_y = int(scn_h / 2 - position[1])   # PsychoPy y-axis is flipped vs EyeLink

    # draw=1        : EyeLink draws the drift target on the Host PC
    # allow_setup=1 : operator can press Esc to go back to camera setup
    try:
        result = el_tracker.doDriftCorrect(el_x, el_y, 1, 1)
    except Exception as e:
        print(f"Drift correction error: {e}")
        return False

    if result == 27:
        # Operator pressed Esc — go to setup menu, then return False so the
        # caller knows to retry drift correction after recalibration.
        try:
            el_tracker.doTrackerSetup()
        except RuntimeError as err:
            print(f"Setup error after drift Esc: {err}")
        return False

    return True


# ==============================================================================
# FIXATION CROSS HELPERS
# ==============================================================================

def _make_fix_cross(win, color, size_pix=FIX_SIZE_PIX, line_width_pix=FIX_LINE_WIDTH_PIX):
    horiz = visual.Line(win,
                        start=(-size_pix / 2, 0), end=(size_pix / 2, 0),
                        lineColor=color, lineWidth=line_width_pix)
    vert  = visual.Line(win,
                        start=(0, -size_pix / 2), end=(0, size_pix / 2),
                        lineColor=color, lineWidth=line_width_pix)
    return [horiz, vert]


def _draw_fix(fix_components):
    for c in fix_components:
        c.draw()


# ==============================================================================
# FACE CLASSIFICATION
# ==============================================================================

def _classify_face(face_path):
    """
    Return (familiarity, source) from the parent folder name.

    Expected layout:
        experimental_faces/uk_celebrities/...
        experimental_faces/israeli_celebrities/...
        experimental_faces/database_faces/...
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


# ==============================================================================
# MAIN TRIAL
# ==============================================================================

def run_trial(el_tracker, win, trial_num, face_image, face_duration_s,
              is_practice=False):
    """
    Single trial: gray fix → face → gray fix → white fix (reproduction) → ITI.

    Returns trial_data dict, None on escape/abort, or 'SKIP' on Shift+S.
    """


    # Drift correction is intentionally skipped before each trial.
    # The session-start calibration is sufficient for this paradigm
    # (face familiarity x duration reproduction). Per-trial drift correction
    # requires operator input on the Host PC for each of 32 trials, which
    # is impractical and unnecessary when gaze position accuracy is not
    # the primary measure. Re-enable drift_correct() here if needed.

    # ----- Build stimuli -----
    face_width_pix, face_height_pix = calculate_face_size()
    _tmp = visual.ImageStim(win, image=face_image)
    img_w, img_h = _tmp.size
    if img_w > 0 and img_h > 0:
        face_size = (face_width_pix, int(face_width_pix * (img_h / img_w)))
    else:
        face_size = (face_width_pix, face_height_pix)
    face_stim  = visual.ImageStim(win, image=face_image, pos=(0, 0), size=face_size)
    fix_gray   = _make_fix_cross(win, FIX_COLOR_GRAY)
    fix_repro  = _make_fix_cross(win, FIX_COLOR_REPRODUCTION,
                                 size_pix=FIX_SIZE_REPRODUCTION_PIX)

    familiarity, face_source = _classify_face(face_image)

    # ----- EyeLink messages & start recording -----
    el_tracker.sendMessage(f"TRIAL_START {trial_num}")
    if is_practice:
        el_tracker.sendMessage("PRACTICE_TRIAL")
    el_tracker.sendMessage(f"FACE_IMAGE {os.path.basename(face_image)}")
    el_tracker.sendMessage(f"FACE_FAMILIARITY {familiarity}")
    el_tracker.sendMessage(f"FACE_SOURCE {face_source}")
    el_tracker.sendMessage(f"FACE_DURATION_PLANNED_MS {int(face_duration_s * 1000)}")

    error = el_tracker.startRecording(1, 1, 1, 1)
    if error:
        print("EyeLink recording error")
        return None
    core.wait(0.1)

    # ===== (a) Pre-face gray fixation =====
    clk = core.Clock(); clk.reset()
    el_tracker.sendMessage("PRE_FACE_FIX_ONSET")
    while clk.getTime() < PRE_FACE_FIX_DURATION:
        check_for_exit()
        if check_for_skip():
            el_tracker.stopRecording(); return 'SKIP'
        _draw_fix(fix_gray)
        win.flip()
        if event.getKeys(keyList=['escape']):
            el_tracker.stopRecording(); return None

    # ===== (b) Face =====
    face_clk = core.Clock(); face_clk.reset()
    el_tracker.sendMessage("FACE_ONSET")
    while face_clk.getTime() < face_duration_s:
        check_for_exit()
        if check_for_skip():
            el_tracker.stopRecording(); return 'SKIP'
        face_stim.draw()
        win.flip()
        if event.getKeys(keyList=['escape']):
            el_tracker.stopRecording(); return None
    actual_face_duration = face_clk.getTime()
    el_tracker.sendMessage("FACE_OFFSET")

    # ===== (c) Post-face gray fixation =====
    post_clk = core.Clock(); post_clk.reset()
    el_tracker.sendMessage("POST_FACE_FIX_ONSET")
    while post_clk.getTime() < POST_FACE_FIX_DURATION:
        check_for_exit()
        if check_for_skip():
            el_tracker.stopRecording(); return 'SKIP'
        _draw_fix(fix_gray)
        win.flip()
        if event.getKeys(keyList=['escape']):
            el_tracker.stopRecording(); return None

    # ===== (d) White fixation — reproduction phase =====
    event.clearEvents()
    repro_clk = core.Clock(); repro_clk.reset()
    el_tracker.sendMessage("REPRODUCTION_ONSET")
    reproduced_duration = None
    while reproduced_duration is None:
        check_for_exit()
        if check_for_skip():
            el_tracker.stopRecording(); return 'SKIP'
        _draw_fix(fix_repro)
        win.flip()
        keys = event.getKeys(keyList=[REPRODUCTION_KEY, 'escape'],
                             timeStamped=repro_clk)
        if keys:
            k_name, k_time = keys[0]
            if k_name == 'escape':
                el_tracker.stopRecording(); return None
            if k_name == REPRODUCTION_KEY:
                reproduced_duration = k_time
                break
    el_tracker.sendMessage(f"REPRODUCTION_RESPONSE {reproduced_duration:.4f}")

    # ===== (e) Blank ITI =====
    iti_clk = core.Clock(); iti_clk.reset()
    el_tracker.sendMessage("ITI_ONSET")
    while iti_clk.getTime() < ITI_DURATION:
        check_for_exit()
        win.flip()
    el_tracker.sendMessage("ITI_OFFSET")

    el_tracker.stopRecording()
    el_tracker.sendMessage(f"TRIAL_END {trial_num}")

    return {
        'trial_num':             trial_num,
        'is_practice':           is_practice,
        'face_image':            face_image,
        'face_familiarity':      familiarity,
        'face_source':           face_source,
        'face_duration_planned_s': face_duration_s,
        'face_duration_actual_s':  actual_face_duration,
        'reproduced_duration_s':   reproduced_duration,
        'reproduction_error_s':    reproduced_duration - face_duration_s,
    }


# ==============================================================================
# PRACTICE
# ==============================================================================

def _show_practice_intro(win, practice_faces):
    fix_small      = _make_fix_cross(win, FIX_COLOR_GRAY)
    fix_large_demo = _make_fix_cross(win, PRACTICE_DEMO_LARGE_CROSS_COLOR,
                                     size_pix=PRACTICE_DEMO_LARGE_CROSS_SIZE_PIX)

    def _line(text, y, height=30, bold=False):
        return visual.TextStim(win, text=text, pos=(0, y), height=height,
                               color='white', bold=bold, wrapWidth=900,
                               font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)

    def _footer(label="Press ENTER to continue.", pos=(0, -300)):
        return visual.TextStim(win, text=label, pos=pos, height=22,
                               color=(0.7, 0.7, 0.7), wrapWidth=900,
                               font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)

    _SEGMENT_WRAP_WIDTH = 4000

    def _measure_width_pix(text, height, bold):
        probe = visual.TextStim(win, text=text, height=height, bold=bold,
                                wrapWidth=_SEGMENT_WRAP_WIDTH,
                                alignText='left', anchorHoriz='left', anchorVert='center',
                                font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)
        return probe.boundingBox[0]

    def _mixed_bold_line(segments, y, height=28):
        widths = [_measure_width_pix(t, h, b) for t, h, b in
                  [(text, height, bold) for text, bold in segments]]
        x = -sum(widths) / 2
        for (text, bold), w in zip(segments, widths):
            visual.TextStim(win, text=text, height=height, bold=bold,
                            color='white', pos=(x, y),
                            wrapWidth=_SEGMENT_WRAP_WIDTH,
                            alignText='left', anchorHoriz='left', anchorVert='center',
                            font=TUTORIAL_FONT_NAME,
                            fontFiles=TUTORIAL_FONT_FILES).draw()
            x += w

    # Screen 1
    _line("In every trial of the experiment, you will see a small\n"
          "fixation cross, followed by a face.", y=220).draw()
    _draw_fix(fix_small)
    _footer().draw()
    win.flip()
    wait_for_keys_or_exit(['space'])

    # Screen 2
    _line("This is how each face will appear.", y=240).draw()
    _mixed_bold_line(
        [("Each face will appear for ", False),
         ("a specific duration", True),
         (" on the screen.", False)], y=180)
    if practice_faces:
        face_width_pix, face_height_pix = calculate_face_size()
        _tmp = visual.ImageStim(win, image=practice_faces[0])
        _w, _h = _tmp.size
        _demo_size = (face_width_pix, int(face_width_pix * (_h / _w))) \
            if _w > 0 and _h > 0 else (face_width_pix, face_height_pix)
        _demo_pos = (0, 0)  # <-- same position as the real trial (run_trial uses pos=(0, 0))

        # Caption directly above the demo image
        caption_y = _demo_pos[1] + _demo_size[1] / 2 + 40
        _line("This is how a face will appear:", y=caption_y, height=26).draw()

        visual.ImageStim(win, image=practice_faces[0],
                         pos=_demo_pos, size=_demo_size).draw()
        footer_y = _demo_pos[1] - _demo_size[1] / 2 - 40
    else:
        footer_y = -300
    _footer(pos=(0, footer_y)).draw()
    win.flip()
    wait_for_keys_or_exit(['space'])

    # Screen 3
    _line("When the face disappears, the small fixation cross will reappear.\n"
          "After, a large fixation cross will appear.", y=260, height=28).draw()
    _mixed_bold_line(
        [("Your task is to match the duration of this ", False),
         ("large cross", True), (" to the", False)], y=180, height=28)
    _line("duration of the face you just saw.", y=140, height=28).draw()
    _mixed_bold_line(
        [("Press the ", False), ("SPACES bar", True),
         (" when you think the ", False), ("same amount of time", True),
         (" has passed.", False)], y=60, height=28)
    _draw_fix(fix_large_demo)
    _footer().draw()
    win.flip()
    wait_for_keys_or_exit(['space'])

    # Screen 4
    _line("A practice trial will start in the next screen. Be ready!", y=0).draw()
    _footer("Press ENTER to begin.").draw()
    win.flip()
    wait_for_keys_or_exit(['space'])


def _show_post_practice_screen(win):
    lines = [
        visual.TextStim(win, text="The practice is over.",
                        pos=(0, 260), height=32, bold=True, color="white",
                        wrapWidth=900, font=TUTORIAL_FONT_NAME,
                        fontFiles=TUTORIAL_FONT_FILES),
        visual.TextStim(win, text="Reminder:",
                        pos=(0, 180), height=28, bold=True, color="white",
                        wrapWidth=900, font=TUTORIAL_FONT_NAME,
                        fontFiles=TUTORIAL_FONT_FILES),
        visual.TextStim(win,
                        text=("Pay attention to how long the face is shown.\n"
                              "Then, when the large cross appears, press the SPACE bar\n"
                              "when you think the same amount of time has passed."),
                        pos=(0, 80), height=26, color="white",
                        wrapWidth=900, font=TUTORIAL_FONT_NAME,
                        fontFiles=TUTORIAL_FONT_FILES),
        visual.TextStim(win,
                        text="Only start timing at the large cross — not during the small crosses.",
                        pos=(0, -60), height=26, bold=True, color="white",
                        wrapWidth=900, font=TUTORIAL_FONT_NAME,
                        fontFiles=TUTORIAL_FONT_FILES),
        visual.TextStim(win,
                        text="The experiment will start in the next screen. Be ready!",
                        pos=(0, -150), height=26, color="white",
                        wrapWidth=900, font=TUTORIAL_FONT_NAME,
                        fontFiles=TUTORIAL_FONT_FILES),
        visual.TextStim(win, text="Press ENTER to begin.",
                        pos=(0, -310), height=22, color=(0.7, 0.7, 0.7),
                        wrapWidth=900, font=TUTORIAL_FONT_NAME,
                        fontFiles=TUTORIAL_FONT_FILES),
    ]
    for stim in lines:
        stim.draw()
    win.flip()
    wait_for_keys_or_exit(['space'])


def run_practice(el_tracker, win, practice_faces):
    _show_practice_intro(win, practice_faces)
    session_count = 0

    while session_count < MAX_PRACTICE_SESSIONS:
        if session_count > 0:
            visual.TextStim(win,
                            text=f"Practice Session {session_count + 1}\n\nPress ENTER to begin.",
                            height=30, wrapWidth=1000).draw()
            win.flip()
            wait_for_keys_or_exit(['space'])

        practice_trials = [(face, dur)
                           for face in practice_faces for dur in FACE_DURATIONS]
        random.shuffle(practice_trials)
        practice_trials = practice_trials[:PRACTICE_TOTAL_TRIALS]

        completed_count = 0
        for trial_idx, (face_img, dur) in enumerate(practice_trials):
            td = run_trial(el_tracker, win,
                           trial_num=f"practice_{session_count}_{trial_idx}",
                           face_image=face_img, face_duration_s=dur,
                           is_practice=True)
            if td == 'SKIP':
                print("[DEBUG] Skipping remaining practice trials.")
                break
            if td is not None:
                completed_count += 1

        if completed_count >= PRACTICE_MIN_CORRECT:
            _show_post_practice_screen(win)
            return True
        else:
            session_count += 1
            if session_count < MAX_PRACTICE_SESSIONS:
                visual.TextStim(win,
                                text=f"You completed {completed_count} of "
                                     f"{PRACTICE_TOTAL_TRIALS} practice trials.\n\n"
                                     "Let's try another practice session.\n\n"
                                     "Press ENTER to continue.",
                                height=30, wrapWidth=1000).draw()
                win.flip()
                wait_for_keys_or_exit(['space'])

    _show_post_practice_screen(win)
    return False


# ==============================================================================
# RATINGS
# ==============================================================================

def run_ratings(win, face_images, participant_id, session):
    from psychopy.visual import Slider

    rating_data = []

    # Intro screen
    visual.TextStim(win,
                    text="In the next part, you will be asked to rate the same images\n"
                         "that you have seen previously.\n\n"
                         "The rating will be based on how familiar you are with each face\n"
                         "and how appealing they were.",
                    pos=(0, 20), height=28, color="white", wrapWidth=900,
                    font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES).draw()
    visual.TextStim(win, text="Press ENTER to continue.",
                    pos=(0, -310), height=22, color=(0.7, 0.7, 0.7), wrapWidth=900,
                    font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES).draw()
    win.flip()
    wait_for_keys_or_exit(["space"])

    win.mouseVisible = True
    event.Mouse(win=win)

    _SW = 600; _SH = 40; _LH = 22; _LC = (0.7, 0.7, 0.7)

    # Practice slider screen
    instr_text    = visual.TextStim(win,
        text="Move the slider to the right if the face felt familiar or appealing.\n"
             "Move it to the left if not.",
        pos=(0, 320), height=26, color="white", wrapWidth=900,
        font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)
    practice_note = visual.TextStim(win, text="Try the sliders below before continuing:",
        pos=(0, 230), height=24, color=(0.8, 0.8, 0.8), wrapWidth=900,
        font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)

    def _mk_slider(y, pos_x=0):
        return Slider(win, ticks=(0, 100), labels=None, pos=(pos_x, y),
                      size=(_SW, _SH), style=["slider"], granularity=1,
                      color="white", fillColor=(0.4, 0.6, 1),
                      borderColor="white", markerColor=(0.4, 0.6, 1),
                      labelColor="white")

    def _mk_label(text, x, y, align):
        return visual.TextStim(win, text=text, pos=(x, y), height=_LH,
                               color=_LC, alignText=align, wrapWidth=100,
                               font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)

    prac_fam_lbl    = visual.TextStim(win, text="How familiar are you with this face?",
                                      pos=(0, 170), height=24, color="white", wrapWidth=900,
                                      font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)
    prac_fam_sl     = _mk_slider(120)
    prac_fam_left   = _mk_label("Completely\nunfamiliar", -_SW/2-110, 120, "right")
    prac_fam_right  = _mk_label("Highly\nfamiliar",      _SW/2+110,  120, "left")
    prac_att_lbl    = visual.TextStim(win, text="How appealing do you find this face?",
                                      pos=(0, 10), height=24, color="white", wrapWidth=900,
                                      font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)
    prac_att_sl     = _mk_slider(-40)
    prac_att_left   = _mk_label("Very\nunattractive", -_SW/2-110, -40, "right")
    prac_att_right  = _mk_label("Very\nattractive",   _SW/2+110,  -40, "left")
    prac_warn       = visual.TextStim(win, text="Please move both sliders before continuing.",
                                      pos=(0, -190), height=22, color=(1, 0.3, 0.3),
                                      wrapWidth=900, font=TUTORIAL_FONT_NAME,
                                      fontFiles=TUTORIAL_FONT_FILES)
    prac_ft_wait    = visual.TextStim(win, text="Move both sliders, then press ENTER to begin rating.",
                                      pos=(0, -310), height=22, color=_LC, wrapWidth=900,
                                      font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)
    prac_ft_ready   = visual.TextStim(win, text="Press ENTER to begin rating.",
                                      pos=(0, -310), height=22, color=(0.4, 1, 0.4),
                                      wrapWidth=900, font=TUTORIAL_FONT_NAME,
                                      fontFiles=TUTORIAL_FONT_FILES)

    prac_show_warn = False
    while True:
        check_for_exit()
        instr_text.draw(); practice_note.draw()
        prac_fam_lbl.draw(); prac_fam_sl.draw()
        prac_fam_left.draw(); prac_fam_right.draw()
        prac_att_lbl.draw(); prac_att_sl.draw()
        prac_att_left.draw(); prac_att_right.draw()
        fam_done = prac_fam_sl.getRating() is not None
        att_done = prac_att_sl.getRating() is not None
        if fam_done and att_done:
            prac_ft_ready.draw(); prac_show_warn = False
        else:
            prac_ft_wait.draw()
            if prac_show_warn: prac_warn.draw()
        win.flip()
        keys = event.getKeys(keyList=["return", "escape"])
        if "escape" in keys: return rating_data
        if check_for_skip(): break
        if "return" in keys:
            if fam_done and att_done: break
            else: prac_show_warn = True

    # Per-face rating
    faces_to_rate = list(face_images)
    random.shuffle(faces_to_rate)
    SW = 600; SH = 40; LH = 22; LC = (0.7, 0.7, 0.7)

    face_width_pix, face_height_pix = calculate_face_size()
    face_stim  = visual.ImageStim(win, image=faces_to_rate[0], pos=(0, 250))
    fam_label  = visual.TextStim(win, text="How familiar are you with this face?",
                                 pos=(0, 120), height=26, color="white", wrapWidth=900,
                                 font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)
    fam_slider = Slider(win, ticks=(0, 100), labels=None, pos=(0, 70),
                        size=(SW, SH), style=["slider"], granularity=1,
                        color="white", fillColor=(0.4, 0.6, 1), borderColor="white",
                        markerColor=(0.4, 0.6, 1), labelColor="white")
    fam_left   = visual.TextStim(win, text="Completely\nunfamiliar",
                                 pos=(-SW/2-110, 70), height=LH, color=LC,
                                 alignText="right", wrapWidth=100,
                                 font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)
    fam_right  = visual.TextStim(win, text="Highly\nfamiliar",
                                 pos=(SW/2+110, 70), height=LH, color=LC,
                                 alignText="left", wrapWidth=100,
                                 font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)
    att_label  = visual.TextStim(win, text="How appealing do you find this face?",
                                 pos=(0, -60), height=26, color="white", wrapWidth=900,
                                 font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)
    att_slider = Slider(win, ticks=(0, 100), labels=None, pos=(0, -110),
                        size=(SW, SH), style=["slider"], granularity=1,
                        color="white", fillColor=(0.4, 0.6, 1), borderColor="white",
                        markerColor=(0.4, 0.6, 1), labelColor="white")
    att_left   = visual.TextStim(win, text="Very\nunattractive",
                                 pos=(-SW/2-110, -110), height=LH, color=LC,
                                 alignText="right", wrapWidth=100,
                                 font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)
    att_right  = visual.TextStim(win, text="Very\nattractive",
                                 pos=(SW/2+110, -110), height=LH, color=LC,
                                 alignText="left", wrapWidth=100,
                                 font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)
    warn_stim  = visual.TextStim(win, text="Please rate both sliders before continuing.",
                                 pos=(0, -300), height=22, color=(1, 0.3, 0.3),
                                 wrapWidth=900, font=TUTORIAL_FONT_NAME,
                                 fontFiles=TUTORIAL_FONT_FILES)
    next_footer= visual.TextStim(win, text="Press ENTER to continue.",
                                 pos=(0, -310), height=22, color=(0.7, 0.7, 0.7),
                                 wrapWidth=900, font=TUTORIAL_FONT_NAME,
                                 fontFiles=TUTORIAL_FONT_FILES)

    skip_ratings = False
    for face_path in faces_to_rate:
        check_for_exit()
        if skip_ratings or check_for_skip():
            print("[DEBUG] Skipping remaining rating trials."); break

        _tmp = visual.ImageStim(win, image=face_path)
        _w, _h = _tmp.size; del _tmp
        fw, fh = calculate_face_size()
        _size = (fw, int(fw * (_h / _w))) if _w > 0 and _h > 0 else (fw, fh)
        MAX_H = 180
        iw, ih = _size
        if ih > MAX_H:
            sc = MAX_H / ih; iw = int(iw * sc); ih = MAX_H
        face_stim.image = face_path
        face_stim.size  = (iw, ih)
        fam_slider.reset(); att_slider.reset()
        show_warning = False

        while True:
            check_for_exit()
            face_stim.draw()
            fam_label.draw(); fam_slider.draw(); fam_left.draw(); fam_right.draw()
            att_label.draw(); att_slider.draw(); att_left.draw(); att_right.draw()
            fam_rated = fam_slider.getRating() is not None
            att_rated = att_slider.getRating() is not None
            if fam_rated and att_rated:
                next_footer.draw(); show_warning = False
            elif show_warning:
                warn_stim.draw()
            win.flip()
            keys = event.getKeys(keyList=["return", "escape"])
            if "escape" in keys: return rating_data
            if check_for_skip(): skip_ratings = True; break
            if "return" in keys:
                if fam_rated and att_rated: break
                else: show_warning = True

        fv = fam_slider.getRating(); av = att_slider.getRating()
        if fv is not None and av is not None:
            rating_data.append({
                "participant_id":       participant_id,
                "session":              session,
                "face_image":           face_path,
                "familiarity_rating":   int(fv),
                "attractiveness_rating":int(av),
            })

    #win.mouseVisible = False
    return rating_data


# ==============================================================================
# POST-TASK QUESTIONNAIRE
# ==============================================================================

def run_post_task_questions(win, participant_id, session):
    from psychopy.visual import Slider
    import string

    SW = 550; SH = 38; LH = 20; LC = (0.7, 0.7, 0.7)
    win.mouseVisible = True
    mouse = event.Mouse(win=win)
    answers = {}

    def q_title(text, y, height=25, bold=False):
        return visual.TextStim(win, text=text, pos=(0, y), height=height,
                               bold=bold, color='white', wrapWidth=860,
                               font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)

    def end_label(text, x, y, align):
        return visual.TextStim(win, text=text, pos=(x, y), height=LH,
                               color=LC, alignText=align,
                               font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)

    _warn_stim = visual.TextStim(win,
                                 text="Please answer all questions before continuing.",
                                 pos=(0, -310), height=20, color=(1, 0.3, 0.3),
                                 wrapWidth=860, font=TUTORIAL_FONT_NAME,
                                 fontFiles=TUTORIAL_FONT_FILES)
    _footer_stim = visual.TextStim(win, text="", pos=(0, -355), height=20,
                                   wrapWidth=860, font=TUTORIAL_FONT_NAME,
                                   fontFiles=TUTORIAL_FONT_FILES)

    def warn(text="Please answer all questions before continuing.", y=-310):
        _warn_stim.text = text; _warn_stim.pos = (0, y); return _warn_stim

    def footer(ready, y=-355):
        _footer_stim.text = "Press ENTER to continue." if ready \
                            else "Answer all questions, then press ENTER."
        _footer_stim.setColor((0.4, 1, 0.4) if ready else LC, 'rgb')
        _footer_stim.pos = (0, y); return _footer_stim

    def make_slider(y):
        return Slider(win, ticks=(0, 100), labels=None, pos=(0, y),
                      size=(SW, SH), style=["slider"], granularity=1,
                      color='white', fillColor=(0.4, 0.6, 1),
                      borderColor='white', markerColor=(0.4, 0.6, 1))

    def make_mc(options, y_top, spacing=38):
        items = []
        for i, opt in enumerate(options):
            y = y_top - i * spacing
            rect  = visual.Rect(win, width=26, height=26, pos=(-SW/2, y),
                                lineColor='white', fillColor=None)
            check = visual.TextStim(win, text=u'\u2714', pos=(-SW/2, y),
                                    height=22, color=(0.4, 0.6, 1), bold=True)
            lbl   = visual.TextStim(win, text=opt, pos=(-SW/2+20, y),
                                    height=22, color='white', alignText='left',
                                    anchorHoriz='left', font=TUTORIAL_FONT_NAME,
                                    fontFiles=TUTORIAL_FONT_FILES)
            items.append([rect, lbl, check, opt])
        return items

    def draw_mc(items, selected_val):
        for rect, lbl, check, val in items:
            rect.setFillColor((-0.6, -0.6, -0.6) if val == selected_val else None, 'rgb')
            rect.draw()
            if val == selected_val: check.draw()
            lbl.draw()

    def check_mc_click(items, selected_val):
        mx, my = mouse.getPos()
        for rect, lbl, check, val in items:
            rx, ry = rect.pos
            hw = rect.width / 2; hh = rect.height / 2
            if rx-hw <= mx <= rx+hw and ry-hh <= my <= ry+hh:
                return val
        return selected_val

    # Screen 0: intro
    visual.TextStim(win, text="Almost done!", pos=(0, 120), height=36,
                    bold=True, color="white", wrapWidth=860,
                    font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES).draw()
    visual.TextStim(win,
                    text="You will now be asked a short questionnaire.\n"
                         "Please answer all questions honestly.",
                    pos=(0, 20), height=28, color="white", wrapWidth=860,
                    font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES).draw()
    visual.TextStim(win, text="Press ENTER to begin.",
                    pos=(0, -310), height=22, color=(0.7, 0.7, 0.7),
                    wrapWidth=860, font=TUTORIAL_FONT_NAME,
                    fontFiles=TUTORIAL_FONT_FILES).draw()
    safe_flip(win)
    wait_for_keys_or_exit(["return"])

    # Screen 1: time, instrument, impulsivity, punctuality
    event.clearEvents(); core.wait(0.1); mouse.clickReset()
    time_q    = q_title("Without looking at the time, how long in minutes have these tasks taken you so far?", y=330)
    instr_q   = q_title("Did you use any instruments to answer the previous question?\nIf so, please clarify.", y=190)
    impuls_q  = q_title("How impulsive are you in general?", y=60)
    impuls_sl = make_slider(10)
    impuls_l  = end_label("Not impulsive\nat all", -SW/2-110, 10, "right")
    impuls_r  = end_label("Very\nimpulsive",       SW/2+110,  10, "left")
    punct_q   = q_title("Are you usually…", y=-90)
    punct_mc  = make_mc(["Early", "Just on time", "Late"], y_top=-125)

    time_buf  = ""
    time_box_rect = visual.Rect(win, width=160, height=36, pos=(0, 275),
                                lineColor='white', fillColor=(-0.8, -0.8, -0.8))
    time_box_txt  = visual.TextStim(win, text="", pos=(0, 275), height=24,
                                    color='white', font=TUTORIAL_FONT_NAME,
                                    fontFiles=TUTORIAL_FONT_FILES)
    instr_buf = ""
    instr_box_rect = visual.Rect(win, width=700, height=36, pos=(0, 130),
                                 lineColor='white', fillColor=(-0.8, -0.8, -0.8))
    instr_box_txt  = visual.TextStim(win, text="", pos=(-340, 130), height=22,
                                     color='white', alignText='left',
                                     anchorHoriz='left', font=TUTORIAL_FONT_NAME,
                                     fontFiles=TUTORIAL_FONT_FILES)
    punct_sel  = None; show_warn1 = False; prev_mouse1 = False; active1 = 'time'

    while True:
        check_for_exit()
        curr_mouse1 = mouse.getPressed()[0]
        if curr_mouse1 and not prev_mouse1:
            if mouse.isPressedIn(time_box_rect):   active1 = 'time'
            elif mouse.isPressedIn(instr_box_rect): active1 = 'instr'
            else:                                   active1 = None
            punct_sel = check_mc_click(punct_mc, punct_sel)
        prev_mouse1 = curr_mouse1
        time_box_rect.setLineColor((0.4, 0.6, 1) if active1 == 'time'  else 'white', 'rgb')
        instr_box_rect.setLineColor((0.4, 0.6, 1) if active1 == 'instr' else 'white', 'rgb')

        enter_pressed1 = False
        keys = event.getKeys(keyList=list(string.digits) + list(string.ascii_letters) +
                             [',', '.', '-', 'backspace', 'space', 'return', 'escape'])
        for k in keys:
            if k == 'escape':
                win.mouseVisible = False; return answers
            elif k == 'return':  enter_pressed1 = True
            elif k == 'space':
                if active1 == 'instr': instr_buf = (instr_buf + ' ')[:200]
            elif k == 'backspace':
                if active1 == 'time': time_buf  = time_buf[:-1]
                else:                 instr_buf = instr_buf[:-1]
            elif k in string.digits:
                if active1 == 'time': time_buf  = (time_buf  + k)[:4]
                else:                 instr_buf = (instr_buf + k)[:200]
            else:
                if active1 == 'instr': instr_buf = (instr_buf + k)[:200]

        time_box_txt.text  = time_buf
        instr_box_txt.text = instr_buf
        time_q.draw();  time_box_rect.draw();  time_box_txt.draw()
        instr_q.draw(); instr_box_rect.draw(); instr_box_txt.draw()
        impuls_q.draw(); impuls_sl.draw(); impuls_l.draw(); impuls_r.draw()
        punct_q.draw(); draw_mc(punct_mc, punct_sel)
        all_done1 = (time_buf != "" and impuls_sl.getRating() is not None
                     and punct_sel is not None)
        footer(all_done1).draw()
        if show_warn1: warn().draw()
        safe_flip(win)
        if enter_pressed1:
            if all_done1:
                answers['time_estimate_min'] = time_buf
                answers['instrument_used']   = instr_buf
                answers['impulsivity']       = int(impuls_sl.getRating())
                answers['punctuality']       = punct_sel
                break
            else: show_warn1 = True

    # Screen 2: face recognition, tiredness, enjoyment
    event.clearEvents(); core.wait(0.1); mouse.clickReset()
    face_rec_q  = q_title("Do you have a problem recognising faces?", y=290)
    face_rec_sl = make_slider(240)
    face_rec_l  = end_label("Very poor at\nrecognising faces", -SW/2-110, 240, "right")
    face_rec_r  = end_label("Very good at\nrecognising faces",  SW/2+110, 240, "left")
    tired_q     = q_title("How tired are you right now?", y=130)
    tired_sl    = make_slider(80)
    tired_l     = end_label("Not at all\ntired", -SW/2-110, 80, "right")
    tired_r     = end_label("Extremely\ntired",  SW/2+110,  80, "left")
    enjoy_q     = q_title("How much did you enjoy this experiment?", y=-80)
    enjoy_sl    = make_slider(-130)
    enjoy_l     = end_label("Not at all", -SW/2-110, -130, "right")
    enjoy_r     = end_label("Very much",  SW/2+110,  -130, "left")
    show_warn2  = False

    while True:
        check_for_exit()
        face_rec_q.draw(); face_rec_sl.draw(); face_rec_l.draw(); face_rec_r.draw()
        tired_q.draw();    tired_sl.draw();    tired_l.draw();    tired_r.draw()
        enjoy_q.draw();    enjoy_sl.draw();    enjoy_l.draw();    enjoy_r.draw()
        all_done2 = (face_rec_sl.getRating() is not None and
                     tired_sl.getRating()    is not None and
                     enjoy_sl.getRating()    is not None)
        footer(all_done2).draw()
        if show_warn2: warn().draw()
        safe_flip(win)
        keys = event.getKeys(keyList=['return', 'escape'])
        if 'escape' in keys: win.mouseVisible = False; return answers
        if 'return' in keys:
            if all_done2:
                answers['face_recognition'] = int(face_rec_sl.getRating())
                answers['tiredness']        = int(tired_sl.getRating())
                answers['enjoyment']        = int(enjoy_sl.getRating())
                break
            else: show_warn2 = True

    # Screen 3: age, gender, country
    event.clearEvents(); core.wait(0.2); safe_flip(win); core.wait(0.1)
    mouse.clickReset()
    age_q       = q_title("How old are you?", y=240)
    age_box_r   = visual.Rect(win, width=160, height=36, pos=(0, 190),
                              lineColor='white', fillColor=(-0.8, -0.8, -0.8))
    age_box_t   = visual.TextStim(win, text="", pos=(0, 190), height=24,
                                  color='white', font=TUTORIAL_FONT_NAME,
                                  fontFiles=TUTORIAL_FONT_FILES)
    gender_q    = q_title("What is your gender?", y=100)
    gender_mc   = make_mc(["Male", "Female", "Other"], y_top=65)
    country_q   = q_title("What country do you currently live in?", y=-60)
    country_box_r = visual.Rect(win, width=700, height=36, pos=(0, -110),
                                lineColor='white', fillColor=(-0.8, -0.8, -0.8))
    country_box_t = visual.TextStim(win, text="", pos=(-340, -110), height=22,
                                    color='white', alignText='left', anchorHoriz='left',
                                    font=TUTORIAL_FONT_NAME, fontFiles=TUTORIAL_FONT_FILES)
    age_buf = ""; country_buf = ""; gender_sel = None
    show_warn3 = False; prev_mouse3 = False; active_field = 'age'

    while True:
        check_for_exit()
        curr_mouse3 = mouse.getPressed()[0]
        if curr_mouse3 and not prev_mouse3:
            if mouse.isPressedIn(age_box_r):     active_field = 'age'
            elif mouse.isPressedIn(country_box_r): active_field = 'country'
            gender_sel = check_mc_click(gender_mc, gender_sel)
        prev_mouse3 = curr_mouse3

        keys3 = event.getKeys(keyList=list(string.digits) + list(string.ascii_letters) +
                              ['-', 'backspace', 'space', 'return', 'escape'])
        enter_pressed = False
        for k in keys3:
            if k == 'escape':
                win.mouseVisible = False; return answers
            elif k == 'return': enter_pressed = True
            elif k == 'space':
                if active_field == 'country': country_buf = (country_buf + ' ')[:100]
            elif k == 'backspace':
                if active_field == 'age': age_buf     = age_buf[:-1]
                else:                     country_buf = country_buf[:-1]
            elif k in string.digits:
                if active_field == 'age':
                    cand = (age_buf + k)[:3]
                    if int(cand) <= 100: age_buf = cand
                else: country_buf = (country_buf + k)[:100]
            else:
                if active_field == 'country': country_buf = (country_buf + k)[:100]

        age_box_t.text     = age_buf
        country_box_t.text = country_buf
        age_box_r.setLineColor((0.4, 0.6, 1) if active_field == 'age'     else 'white', 'rgb')
        country_box_r.setLineColor((0.4, 0.6, 1) if active_field == 'country' else 'white', 'rgb')

        age_valid = False; age_warn_msg = ""
        if age_buf != "":
            ai = int(age_buf)
            if ai < 18:   age_warn_msg = "Age must be 18 or older."
            elif ai > 100: age_warn_msg = "Age must be 100 or younger."
            else:          age_valid = True

        all_done3 = (age_valid and gender_sel is not None
                     and country_buf.strip() != "")
        age_q.draw(); age_box_r.draw(); age_box_t.draw()
        if age_warn_msg:
            _warn_stim.text = age_warn_msg; _warn_stim.pos = (0, 158)
            _warn_stim.setColor((1, 0.3, 0.3), 'rgb'); _warn_stim.draw()
            _warn_stim.pos = (0, -310)
        gender_q.draw(); draw_mc(gender_mc, gender_sel)
        country_q.draw(); country_box_r.draw(); country_box_t.draw()
        footer(all_done3).draw()
        if show_warn3: warn().draw()
        safe_flip(win)
        if enter_pressed:
            if all_done3:
                answers['age']     = str(int(age_buf))
                answers['gender']  = gender_sel
                answers['country'] = country_buf.strip()
                break
            else: show_warn3 = True

    win.mouseVisible = False
    return answers


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    verify_display_parameters()

    # Arm the emergency-kill watchdog (Ctrl+Alt+Q) before anything else, so
    # it's active even if the window/tracker setup itself hangs.
    # watchdog_thread = threading.Thread(target=_emergency_kill_watchdog, daemon=True)
    # watchdog_thread.start()

    exp_info = {'Participant ID': '', 'Session': '001'}
    dlg = gui.DlgFromDict(dictionary=exp_info, title='Face Familiarity Experiment')
    if not dlg.OK:
        core.quit()

    # Anchor data folder to the script location so it's always predictable.
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(_script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    edf_fname = f"{exp_info['Participant ID'][:4]}{exp_info['Session']}.edf"

    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    behavioral_fname = os.path.join(
        data_dir,
        f"{exp_info['Participant ID']}_sess{exp_info['Session']}_{ts}_behavioral.csv")

    # ----- Propixx -----
    propixx_restore = init_propixx_normal_mode()
    import atexit
    atexit.register(propixx_restore)

    # ----- Window -----
    use_fullscreen = (not DEBUG_MODE) and (not WINDOWED_MODE)

    if use_fullscreen and USE_PROPIXX and _HAVE_PROPIXX_MODULE:
        win_size     = fl.FULL_RES
        monitor_name = 'propixx_tester'
    elif use_fullscreen:
        win_size     = (SCREEN_WIDTH, SCREEN_HEIGHT)
        monitor_name = None
    else:
        win_size     = WINDOWED_SIZE
        monitor_name = None

    print(f"Display mode: {'fullscreen' if use_fullscreen else 'windowed'}, "
          f"size={win_size}, debug={DEBUG_MODE}, windowed_flag={WINDOWED_MODE}")

    win_kwargs = dict(
        size=win_size,
        fullscr=use_fullscreen,
        screen=0,
        allowGUI=True,          # required for EyeLinkCoreGraphicsPsychoPy
        color=[-1, -1, -1],
        units='pix',
    )
    if monitor_name is not None:
        win_kwargs['monitor'] = monitor_name

    win = visual.Window(**win_kwargs)
    win.mouseVisible = False

    try:
        fps = win.getActualFrameRate(nIdentical=10, nMaxFrames=120,
                                     nWarmUpFrames=10, threshold=1.0)
        print(f"Measured refresh rate: {fps:.2f} Hz "
              f"(expected ~120 Hz on Propixx normal mode).")
    except Exception as e:
        print(f"Frame-rate check failed: {e}")

    # ----- EyeLink -----
    el_tracker = setup_eyelink(win, edf_fname)
    run_calibration(el_tracker, win)

    # ----- Load stimuli -----
    practice_face_folder    = os.path.join(_script_dir, "practice_faces")
    experimental_face_folder= os.path.join(_script_dir, "experimental_faces")

    practice_faces = []
    if os.path.exists(practice_face_folder):
        practice_faces = [
            os.path.join(practice_face_folder, f)
            for f in os.listdir(practice_face_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
    if not practice_faces:
        print(f"Warning: no practice faces found in '{practice_face_folder}'")
        el_tracker.close(); win.close(); core.quit()

    experimental_faces = []
    if os.path.exists(experimental_face_folder):
        for root, dirs, files in os.walk(experimental_face_folder):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    experimental_faces.append(os.path.join(root, f))
    if not experimental_faces:
        print(f"Warning: no experimental faces found in '{experimental_face_folder}'")
        print("Expected subfolders: uk_celebrities/, israeli_celebrities/, database_faces/")
        el_tracker.close(); win.close(); core.quit()

    # ----- Practice -----
    all_behavioral_data = []
    print("Running practice trials...")
    run_practice(el_tracker, win, practice_faces)

    # ----- Build trial list (balanced durations) -----
    faces_shuffled = list(experimental_faces)
    random.shuffle(faces_shuffled)
    n = N_EXPERIMENTAL_TRIALS // 2
    trial_face_list = (
        [(face, FACE_DURATIONS[0]) for face in faces_shuffled[:n]] +
        [(face, FACE_DURATIONS[1]) for face in faces_shuffled[n:n*2]]
    )
    random.shuffle(trial_face_list)

    # ----- Main experiment -----
    for trial_num, (face_img, dur) in enumerate(trial_face_list):
        td = run_trial(el_tracker, win,
                       trial_num=trial_num,
                       face_image=face_img,
                       face_duration_s=dur,
                       is_practice=False)
        if td == 'SKIP':
            print("[DEBUG] Skipping remaining experiment trials."); break
        if td is not None:
            td['participant_id'] = exp_info['Participant ID']
            td['session']        = exp_info['Session']
            all_behavioral_data.append(td)
        check_for_exit()
        if 'escape' in event.getKeys():
            break

    # ----- Wrap up EyeLink -----
    el_tracker.sendMessage("EXPERIMENT_END")
    el_tracker.setOfflineMode()
    core.wait(0.1)
    el_tracker.closeDataFile()

    # ----- Save behavioural CSV -----
    if all_behavioral_data:
        fieldnames = ['participant_id', 'session', 'trial_num', 'is_practice',
                      'face_image', 'face_familiarity', 'face_source',
                      'face_duration_planned_s', 'face_duration_actual_s',
                      'reproduced_duration_s', 'reproduction_error_s']
        with open(behavioral_fname, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_behavioral_data)
        print(f"Behavioural data saved: {behavioral_fname}")

    # ----- Ratings -----
    print("Entering rating phase.")
    rated_faces = list({td['face_image'] for td in all_behavioral_data}) \
                  or list(experimental_faces)
    print(f"Rating {len(rated_faces)} faces.")
    rating_results = run_ratings(win, rated_faces,
                                 exp_info['Participant ID'], exp_info['Session'])
    if rating_results:
        ratings_fname = os.path.join(
            data_dir,
            f"{exp_info['Participant ID']}_sess{exp_info['Session']}_{ts}_ratings.csv")
        with open(ratings_fname, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['participant_id', 'session', 'face_image',
                               'familiarity_rating', 'attractiveness_rating'])
            writer.writeheader()
            writer.writerows(rating_results)
        print(f"Rating data saved: {ratings_fname}")

    # ----- Post-task questionnaire -----
    print("Entering post-task questions.")
    event.clearEvents()
    for _ in range(5):
        try: win.flip()
        except Exception: pass
        core.wait(0.1)
    event.clearEvents()
    post_answers = run_post_task_questions(win, exp_info['Participant ID'],
                                           exp_info['Session'])
    if post_answers:
        post_fname = os.path.join(
            data_dir,
            f"{exp_info['Participant ID']}_sess{exp_info['Session']}_{ts}_post_task.csv")
        post_answers['participant_id'] = exp_info['Participant ID']
        post_answers['session']        = exp_info['Session']
        fieldnames_post = ['participant_id', 'session', 'time_estimate_min',
                           'instrument_used', 'impulsivity', 'punctuality',
                           'face_recognition', 'tiredness', 'enjoyment',
                           'age', 'gender', 'country']
        with open(post_fname, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_post, extrasaction='ignore')
            writer.writeheader()
            writer.writerow(post_answers)
        print(f"Post-task data saved: {post_fname}")

    # ----- Transfer EDF -----
    local_edf = os.path.join(
        data_dir,
        f"{exp_info['Participant ID']}_{exp_info['Session']}.edf")
    try:
        el_tracker.receiveDataFile(edf_fname, local_edf)
        print(f"EDF saved: {local_edf}")
    except RuntimeError as error:
        print(f"EDF transfer error: {error}")

    el_tracker.close()

    # ----- Thank you -----
    visual.TextStim(win,
                    text="Thank you for participating!\n\nPress any key to exit.",
                    height=30).draw()
    win.flip()
    event.waitKeys()

    win.close()
    core.quit()


if __name__ == "__main__":
    main()