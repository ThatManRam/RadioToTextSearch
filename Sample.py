#!/usr/bin/env python3
import math
import time
import signal
import threading
from collections import deque
from fractions import Fraction

import numpy as np
import sounddevice as sd
from rtlsdr import RtlSdr
from scipy.signal import firwin, lfilter, resample_poly

# ============================================================
# USER SETTINGS (initial)
# ============================================================
STATION_FREQ_HZ = 90_098_000      # 90.100 MHz (what you want to hear)
TUNE_OFFSET_HZ  = 000         # -707 kHz shown in SDR++ (station is below center)
CENTER_FREQ_HZ  = STATION_FREQ_HZ + TUNE_OFFSET_HZ  # 90.807 MHz hardware freq

RF_SAMPLE_RATE  = 1_024_000
IF_RATE         = 240_000
AUDIO_RATE      = 48_000

TUNER_GAIN      = 40              # try 10..40 if overload/IMD
FREQ_PPM        = 0               # set to your dongle PPM if known (e.g., 20)

DEEMPH_TAU      = 75e-6           # 75us (US). Use 50e-6 in many other regions.
AUDIO_GAIN_DB   = -12.3

# Playback / buffering
AUDIO_BLOCK_FRAMES = 1024
IQ_CHUNK_SAMPLES   = 262_144       # try 131072 if CPU constrained
MAX_BUFFER_CHUNKS  = 120
TRIM_TO_CHUNKS     = 40

# Filter width (adjustable while running)
WFM_CHAN_CUTOFF_HZ_INIT = 100_000  # start at "Normal-ish"

# ============================================================
# DSP helpers
# ============================================================
def lpf(fs, cutoff_hz, taps=129):
    return firwin(taps, cutoff_hz, fs=fs)

def bpf(fs, lo_hz, hi_hz, taps=257):
    return firwin(taps, [lo_hz, hi_hz], pass_zero=False, fs=fs)

def fir_filter(x, taps, state):
    y, zf = lfilter(taps, 1.0, x, zi=state)
    return y, zf

def rational_resample_ratio(fs_in, fs_out, max_den=512):
    frac = Fraction(fs_out, fs_in).limit_denominator(max_den)
    return frac.numerator, frac.denominator

class Deemphasis:
    # 1-pole deemphasis: y[n] = (1-a)*x[n] + a*y[n-1]
    def __init__(self, fs, tau):
        self.a = float(np.exp(-1.0 / (fs * tau)))
        self.b = [1.0 - self.a]
        self.a_den = [1.0, -self.a]
        self.zi = np.array([0.0], dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        y, zf = lfilter(self.b, self.a_den, x.astype(np.float32, copy=False), zi=self.zi)
        self.zi = zf
        return y.astype(np.float32)

class AudioAGC:
    """
    Moderate ("Medium-ish") AGC on stereo program audio.
    """
    def __init__(self, fs=AUDIO_RATE, target=0.12, attack_ms=25, release_ms=250):
        self.target = float(target)
        self.attack_a = math.exp(-1.0 / (fs * (attack_ms / 1000.0)))
        self.release_a = math.exp(-1.0 / (fs * (release_ms / 1000.0)))
        self.env = 1e-3

    def process(self, stereo: np.ndarray) -> np.ndarray:
        if stereo.size == 0:
            return stereo

        env_in = np.max(np.abs(stereo), axis=1)
        for v in env_in:
            a = self.attack_a if v > self.env else self.release_a
            self.env = a * self.env + (1.0 - a) * float(v)

        gain = self.target / (self.env + 1e-9)
        gain = float(np.clip(gain, 0.1, 10.0))
        return (stereo * gain).astype(np.float32)

# ============================================================
# WFM Stereo Demod (with offset tuning + live-adjustable filter)
# ============================================================
class WFMStereoOffset:
    def __init__(self, chan_cutoff_hz: int):
        self.cfg_lock = threading.Lock()

        # Adjustable IQ channel select filter
        self._build_chan_filter(chan_cutoff_hz)

        # Composite filters at IF_RATE
        self.mono_taps = lpf(IF_RATE, 15_000, taps=129)
        self.mono_state = np.zeros(len(self.mono_taps) - 1, dtype=np.float32)

        self.pilot_taps = bpf(IF_RATE, 18_500, 19_500, taps=257)
        self.pilot_state = np.zeros(len(self.pilot_taps) - 1, dtype=np.float32)

        self.sub_taps = bpf(IF_RATE, 36_000, 40_000, taps=257)
        self.sub_state = np.zeros(len(self.sub_taps) - 1, dtype=np.float32)

        self.stereo_taps = bpf(IF_RATE, 23_000, 53_000, taps=257)
        self.stereo_state = np.zeros(len(self.stereo_taps) - 1, dtype=np.float32)

        self.lr_lpf_taps = lpf(IF_RATE, 15_000, taps=129)
        self.lr_lpf_state = np.zeros(len(self.lr_lpf_taps) - 1, dtype=np.float32)

        # FM discriminator continuity
        self.prev_iq = None

        # NCO for offset tuning
        self.phase = 0.0
        self.step = 2.0 * np.pi * (TUNE_OFFSET_HZ / RF_SAMPLE_RATE)

        # Resample ratios
        self.up_if, self.down_if = rational_resample_ratio(RF_SAMPLE_RATE, IF_RATE)
        self.up_a, self.down_a = rational_resample_ratio(IF_RATE, AUDIO_RATE)

        # Audio conditioning
        self.deemph_L = Deemphasis(AUDIO_RATE, DEEMPH_TAU)
        self.deemph_R = Deemphasis(AUDIO_RATE, DEEMPH_TAU)
        self.agc = AudioAGC(fs=AUDIO_RATE)

        self.audio_gain = float(10 ** (AUDIO_GAIN_DB / 20.0))

    def _build_chan_filter(self, cutoff_hz: int):
        cutoff_hz = int(np.clip(cutoff_hz, 40_000, 160_000))
        self.chan_cutoff_hz = cutoff_hz
        self.chan_taps = lpf(RF_SAMPLE_RATE, cutoff_hz, taps=129)
        self.chan_state = np.zeros(len(self.chan_taps) - 1, dtype=np.complex64)

    def set_filter_width(self, cutoff_hz: int):
        with self.cfg_lock:
            self._build_chan_filter(cutoff_hz)

    def adjust_filter_width(self, delta_hz: int):
        with self.cfg_lock:
            self._build_chan_filter(self.chan_cutoff_hz + delta_hz)

    def get_filter_width(self) -> int:
        with self.cfg_lock:
            return int(self.chan_cutoff_hz)

    def mix_offset_to_baseband(self, iq: np.ndarray) -> np.ndarray:
        n = np.arange(len(iq), dtype=np.float32)
        ph = self.phase + self.step * n
        osc = np.exp(1j * ph).astype(np.complex64)
        self.phase = float((self.phase + self.step * len(iq)) % (2.0 * np.pi))
        return (iq.astype(np.complex64, copy=False) * osc)

    def fm_discriminator(self, iq_if: np.ndarray) -> np.ndarray:
        if self.prev_iq is None:
            self.prev_iq = iq_if[0]
        x_prev = np.concatenate(([self.prev_iq], iq_if[:-1]))
        self.prev_iq = iq_if[-1]
        d = iq_if * np.conj(x_prev)
        return np.angle(d).astype(np.float32)

    def process_block(self, iq: np.ndarray) -> np.ndarray:
        # 0) Offset mix
        iq = self.mix_offset_to_baseband(iq)

        # 1) Channel select (lock so taps/state can't change mid-filter)
        with self.cfg_lock:
            iq_f, self.chan_state = fir_filter(iq, self.chan_taps, self.chan_state)

        # 2) Resample IQ -> IF_RATE
        iq_if = resample_poly(iq_f, self.up_if, self.down_if).astype(np.complex64)

        # 3) FM discriminator -> composite
        fm = self.fm_discriminator(iq_if)

        # 4) L+R
        mono, self.mono_state = fir_filter(fm, self.mono_taps, self.mono_state)

        # 5) Pilot
        pilot, self.pilot_state = fir_filter(fm, self.pilot_taps, self.pilot_state)

        # 6) Regenerate 38k
        sub38_raw = pilot * pilot
        sub38, self.sub_state = fir_filter(sub38_raw, self.sub_taps, self.sub_state)
        rms = float(np.sqrt(np.mean(sub38 * sub38)) + 1e-9)
        sub38 = sub38 / rms

        # 7) Stereo band -> mix down
        stereo_band, self.stereo_state = fir_filter(fm, self.stereo_taps, self.stereo_state)
        lr = stereo_band * (2.0 * sub38)
        lr, self.lr_lpf_state = fir_filter(lr, self.lr_lpf_taps, self.lr_lpf_state)

        # 8) Reconstruct L/R
        left = 0.5 * (mono + lr)
        right = 0.5 * (mono - lr)

        # 9) Resample to audio
        left_a = resample_poly(left, self.up_a, self.down_a).astype(np.float32)
        right_a = resample_poly(right, self.up_a, self.down_a).astype(np.float32)

        # 10) Deemphasis
        left_a = self.deemph_L.process(left_a)
        right_a = self.deemph_R.process(right_a)

        stereo = np.column_stack([left_a, right_a]).astype(np.float32)

        # 11) AGC
        stereo = self.agc.process(stereo)

        # 12) Audio gain
        stereo *= self.audio_gain

        # 13) Clip protection
        peak = float(np.max(np.abs(stereo)) + 1e-9)
        if peak > 0.98:
            stereo *= (0.98 / peak)

        return stereo

# ============================================================
# Keyboard control (Linux terminal): single-key controls
# ============================================================
def start_key_thread(stop_event: threading.Event, demod: WFMStereoOffset, request_stop_fn):
    import sys
    import select

    def print_help():
        print(
            "\nControls:\n"
            "  1 = filter 60 kHz\n"
            "  2 = filter 80 kHz\n"
            "  3 = filter 100 kHz\n"
            "  4 = filter 120 kHz\n"
            "  [ = -5 kHz filter\n"
            "  ] = +5 kHz filter\n"
            "  ? = help\n"
            "  q = quit\n"
        )

    def worker():
        # If not a TTY (e.g., some IDE consoles), fall back silently
        if not sys.stdin.isatty():
            print("Keyboard controls: stdin is not a TTY; run from a normal terminal for live keys.")
            return

        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        print_help()
        print(f"Current filter cutoff: {demod.get_filter_width()/1000:.1f} kHz")

        try:
            while not stop_event.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = sys.stdin.read(1)

                if ch == "q":
                    request_stop_fn()
                    break
                elif ch == "?":
                    print_help()
                    print(f"Current filter cutoff: {demod.get_filter_width()/1000:.1f} kHz")
                elif ch == "1":
                    demod.set_filter_width(60_000)
                    print(f"Filter cutoff set to {demod.get_filter_width()/1000:.1f} kHz")
                elif ch == "2":
                    demod.set_filter_width(80_000)
                    print(f"Filter cutoff set to {demod.get_filter_width()/1000:.1f} kHz")
                elif ch == "3":
                    demod.set_filter_width(100_000)
                    print(f"Filter cutoff set to {demod.get_filter_width()/1000:.1f} kHz")
                elif ch == "4":
                    demod.set_filter_width(120_000)
                    print(f"Filter cutoff set to {demod.get_filter_width()/1000:.1f} kHz")
                elif ch == "[":
                    demod.adjust_filter_width(-5_000)
                    print(f"Filter cutoff: {demod.get_filter_width()/1000:.1f} kHz")
                elif ch == "]":
                    demod.adjust_filter_width(+5_000)
                    print(f"Filter cutoff: {demod.get_filter_width()/1000:.1f} kHz")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t

# ============================================================
# Main: continuous audio + robust Ctrl+C + live filter width
# ============================================================
def main():
    print(f"Station:  {STATION_FREQ_HZ/1e6:.3f} MHz")
    print(f"Hardware: {CENTER_FREQ_HZ/1e6:.6f} MHz (offset -{TUNE_OFFSET_HZ/1e3:.3f} kHz)")
    print(f"Mode: WFM (stereo), AGC: Medium-ish, Audio gain: {AUDIO_GAIN_DB} dB")
    print("Press Ctrl+C (or 'q') to stop.\n")

    stop_event = threading.Event()

    # Audio buffer between SDR (producer) and audio callback (consumer)
    buf_lock = threading.Lock()
    buf = deque()
    leftover = np.zeros((0, 2), dtype=np.float32)

    demod = WFMStereoOffset(WFM_CHAN_CUTOFF_HZ_INIT)

    sdr = RtlSdr()
    sdr.sample_rate = RF_SAMPLE_RATE
    sdr.center_freq = CENTER_FREQ_HZ
    sdr.gain = TUNER_GAIN

    if FREQ_PPM:
        sdr.freq_correction = int(FREQ_PPM)

    def request_stop(*_args):
        stop_event.set()
        try:
            sdr.cancel_read_async()
        except Exception:
            pass

    # Ctrl+C handling
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def rtl_callback(iq, _ctx):
        if stop_event.is_set():
            return
        try:
            stereo = demod.process_block(iq)

            with buf_lock:
                buf.append(stereo)
                # Bound latency: if we fall behind, drop old chunks
                if len(buf) > MAX_BUFFER_CHUNKS:
                    while len(buf) > TRIM_TO_CHUNKS:
                        buf.popleft()
        except Exception:
            # keep capture alive even if one block fails
            pass

    def audio_callback(outdata, frames, _time, status):
        nonlocal leftover

        out = np.zeros((frames, 2), dtype=np.float32)
        need = frames
        idx = 0

        if len(leftover) > 0:
            take = min(need, len(leftover))
            out[idx:idx + take] = leftover[:take]
            leftover = leftover[take:]
            idx += take
            need -= take

        while need > 0:
            with buf_lock:
                chunk = buf.popleft() if buf else None
            if chunk is None:
                break

            take = min(need, len(chunk))
            out[idx:idx + take] = chunk[:take]
            idx += take
            need -= take

            if take < len(chunk):
                leftover = chunk[take:]
                break

        outdata[:] = out

    # Start keyboard controls
    start_key_thread(stop_event, demod, request_stop)

    # Start audio stream
    stream = sd.OutputStream(
        samplerate=AUDIO_RATE,
        channels=2,
        dtype="float32",
        blocksize=AUDIO_BLOCK_FRAMES,
        callback=audio_callback,
        latency="low",
    )
    stream.start()

    # Run RTL async in background so main thread remains signal-responsive
    reader_thread = threading.Thread(
        target=lambda: sdr.read_samples_async(rtl_callback, IQ_CHUNK_SAMPLES),
        daemon=True
    )
    reader_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        request_stop()
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        try:
            sdr.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
