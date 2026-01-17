#!/usr/bin/env python3
import math
import time
import signal
import threading
import queue
from collections import deque
from fractions import Fraction

import numpy as np
import sounddevice as sd
from rtlsdr import RtlSdr
from scipy.signal import firwin, lfilter, resample_poly

import whisper  # pip install openai-whisper

# ============================================================
# USER SETTINGS
# ============================================================
STATION_FREQ_HZ = 90_098_000
TUNE_OFFSET_HZ  = 0
CENTER_FREQ_HZ  = STATION_FREQ_HZ + TUNE_OFFSET_HZ

RF_SAMPLE_RATE  = 1_024_000
IF_RATE         = 240_000
AUDIO_RATE      = 48_000

TUNER_GAIN      = 25
FREQ_PPM        = 0  # set non-zero only if you know your ppm; some drivers reject 0-set

DEEMPH_TAU      = 75e-6
AUDIO_GAIN_DB   = -12.3

AUDIO_BLOCK_FRAMES = 1024
IQ_CHUNK_SAMPLES   = 262_144
MAX_BUFFER_CHUNKS  = 120
TRIM_TO_CHUNKS     = 40

WFM_CHAN_CUTOFF_HZ_INIT = 100_000

# STT settings
STT_MODEL_NAME = "base"   # "tiny", "base", "small", "medium", "large"
STT_TARGET_SR  = 16_000
STT_CHUNK_SEC  = 8.0      # transcribe every ~8 seconds of audio
STT_MIN_RMS    = 0.015    # simple "speech present" gate; tweak (0.01-0.03)
STT_QUEUE_MAX  = 200      # frames queued from audio callback

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
# WFM Stereo Demod (as in your file)
# ============================================================
class WFMStereoOffset:
    def __init__(self, chan_cutoff_hz: int):
        self.cfg_lock = threading.Lock()
        self._build_chan_filter(chan_cutoff_hz)

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

        self.prev_iq = None

        self.phase = 0.0
        self.step = 2.0 * np.pi * (TUNE_OFFSET_HZ / RF_SAMPLE_RATE)

        self.up_if, self.down_if = rational_resample_ratio(RF_SAMPLE_RATE, IF_RATE)
        self.up_a, self.down_a = rational_resample_ratio(IF_RATE, AUDIO_RATE)

        self.deemph_L = Deemphasis(AUDIO_RATE, DEEMPH_TAU)
        self.deemph_R = Deemphasis(AUDIO_RATE, DEEMPH_TAU)
        self.agc = AudioAGC(fs=AUDIO_RATE)

        self.audio_gain = float(10 ** (AUDIO_GAIN_DB / 20.0))

    def _build_chan_filter(self, cutoff_hz: int):
        cutoff_hz = int(np.clip(cutoff_hz, 40_000, 160_000))
        self.chan_cutoff_hz = cutoff_hz
        self.chan_taps = lpf(RF_SAMPLE_RATE, cutoff_hz, taps=129)
        self.chan_state = np.zeros(len(self.chan_taps) - 1, dtype=np.complex64)

    def mix_offset_to_baseband(self, iq: np.ndarray) -> np.ndarray:
        if TUNE_OFFSET_HZ == 0:
            return iq.astype(np.complex64, copy=False)
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
        iq = self.mix_offset_to_baseband(iq)

        with self.cfg_lock:
            iq_f, self.chan_state = fir_filter(iq, self.chan_taps, self.chan_state)

        iq_if = resample_poly(iq_f, self.up_if, self.down_if).astype(np.complex64)
        fm = self.fm_discriminator(iq_if)

        mono, self.mono_state = fir_filter(fm, self.mono_taps, self.mono_state)
        pilot, self.pilot_state = fir_filter(fm, self.pilot_taps, self.pilot_state)

        sub38_raw = pilot * pilot
        sub38, self.sub_state = fir_filter(sub38_raw, self.sub_taps, self.sub_state)
        rms = float(np.sqrt(np.mean(sub38 * sub38)) + 1e-9)
        sub38 = sub38 / rms

        stereo_band, self.stereo_state = fir_filter(fm, self.stereo_taps, self.stereo_state)
        lr = stereo_band * (2.0 * sub38)
        lr, self.lr_lpf_state = fir_filter(lr, self.lr_lpf_taps, self.lr_lpf_state)

        left = 0.5 * (mono + lr)
        right = 0.5 * (mono - lr)

        left_a = resample_poly(left, self.up_a, self.down_a).astype(np.float32)
        right_a = resample_poly(right, self.up_a, self.down_a).astype(np.float32)

        left_a = self.deemph_L.process(left_a)
        right_a = self.deemph_R.process(right_a)

        stereo = np.column_stack([left_a, right_a]).astype(np.float32)
        stereo = self.agc.process(stereo)
        stereo *= self.audio_gain

        peak = float(np.max(np.abs(stereo)) + 1e-9)
        if peak > 0.98:
            stereo *= (0.98 / peak)

        return stereo

# ============================================================
# STT worker: consumes audio from queue, transcribes, prints
# ============================================================
def start_stt_thread(stop_event: threading.Event, stt_q: queue.Queue):
    model = whisper.load_model(STT_MODEL_NAME)

    # accumulate mono at 48k then resample to 16k for Whisper
    buf = np.zeros((0,), dtype=np.float32)
    target_samples = int(STT_CHUNK_SEC * AUDIO_RATE)

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x * x)) + 1e-12)

    def worker():
        nonlocal buf
        print(f"STT: Whisper model '{STT_MODEL_NAME}' loaded. Transcribing ~{STT_CHUNK_SEC:.1f}s chunks.\n")
        while not stop_event.is_set():
            try:
                mono_chunk = stt_q.get(timeout=0.2)  # mono float32 at 48k
            except queue.Empty:
                continue

            buf = np.concatenate([buf, mono_chunk])

            if len(buf) >= target_samples:
                segment = buf[:target_samples]
                buf = buf[target_samples:]

                # simple speech gate
                if rms(segment) < STT_MIN_RMS:
                    continue

                # resample 48k -> 16k
                seg_16k = resample_poly(segment, STT_TARGET_SR, AUDIO_RATE).astype(np.float32)

                # transcribe
                try:
                    result = model.transcribe(seg_16k, language="en", fp16=False)
                    text = (result.get("text") or "").strip()
                    if text:
                        print(text)
                except Exception as e:
                    print(f"STT error: {e}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t

# ============================================================
# Main: SDR -> demod -> (1) speaker stream (2) STT queue
# ============================================================
def main():
    print(f"Tuning: {STATION_FREQ_HZ/1e6:.3f} MHz (center {CENTER_FREQ_HZ/1e6:.6f} MHz)")
    print("Audio: continuous playback + STT in background. Ctrl+C to stop.\n")

    stop_event = threading.Event()

    # Buffer for audio playback
    buf_lock = threading.Lock()
    buf = deque()
    leftover = np.zeros((0, 2), dtype=np.float32)

    # Queue for STT (mono)
    stt_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=STT_QUEUE_MAX)

    demod = WFMStereoOffset(WFM_CHAN_CUTOFF_HZ_INIT)

    sdr = RtlSdr()
    sdr.sample_rate = RF_SAMPLE_RATE
    sdr.center_freq = CENTER_FREQ_HZ
    sdr.gain = TUNER_GAIN

    # Avoid setting 0 ppm (some libs reject it)
    if FREQ_PPM:
        try:
            sdr.freq_correction = int(FREQ_PPM)
        except Exception as e:
            print(f"Warning: could not set freq_correction={FREQ_PPM} ppm: {e}")

    def request_stop(*_args):
        stop_event.set()
        try:
            sdr.cancel_read_async()
        except Exception:
            pass

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    # Start STT thread
    start_stt_thread(stop_event, stt_q)

    def rtl_callback(iq, _ctx):
        if stop_event.is_set():
            return
        try:
            stereo = demod.process_block(iq)

            # playback buffer
            with buf_lock:
                buf.append(stereo)
                if len(buf) > MAX_BUFFER_CHUNKS:
                    while len(buf) > TRIM_TO_CHUNKS:
                        buf.popleft()

            # STT tap: downmix to mono and enqueue (non-blocking)
            mono = stereo.mean(axis=1).astype(np.float32, copy=False)
            try:
                stt_q.put_nowait(mono)
            except queue.Full:
                # drop STT frames if we're behind; do not impact playback
                pass

        except Exception:
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

    # Audio stream
    stream = sd.OutputStream(
        samplerate=AUDIO_RATE,
        channels=2,
        dtype="float32",
        blocksize=AUDIO_BLOCK_FRAMES,
        callback=audio_callback,
        latency="low",
    )
    stream.start()

    # SDR async reader in background thread
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
