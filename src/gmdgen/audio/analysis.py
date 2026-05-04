from __future__ import annotations

import math
import statistics
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AudioBuffer:
    samples: list[float]
    sample_rate: int

    @property
    def duration(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / self.sample_rate


@dataclass(slots=True)
class FrameFeature:
    time: float
    rms: float
    onset_strength: float
    spectral_flux: float
    mel_bins: list[float] = field(default_factory=list)


@dataclass(slots=True)
class BeatFeature:
    index: int
    time: float
    strength: float
    is_downbeat: bool
    local_energy: float
    rhythmic_density: float


@dataclass(slots=True)
class OnsetFeature:
    time: float
    strength: float


@dataclass(slots=True)
class AudioConfidenceReport:
    overall: float
    bpm_confidence: float
    beat_confidence: float
    onset_confidence: float
    section_confidence: float
    tempo_stability: float
    backend: str
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SectionFeature:
    section_id: int
    start_time: float
    end_time: float
    section_type: str
    mean_energy: float
    rhythmic_density: float
    confidence: float
    energy_peak: float = 0.0

    @property
    def energy_mean(self) -> float:
        return self.mean_energy


@dataclass(slots=True)
class AudioFeatures:
    duration: float
    bpm: float
    beat_times: list[float]
    downbeat_times: list[float]
    onset_times: list[float]
    rms_envelope: list[tuple[float, float]]
    onset_envelope: list[tuple[float, float]]
    spectral_flux: list[tuple[float, float]]
    mel_spectrogram: list[list[float]]
    chroma: list[list[float]]
    frame_features: list[FrameFeature]
    beat_features: list[BeatFeature]
    sections: list[SectionFeature]
    low_energy_sections: list[tuple[float, float]]
    drop_points: list[float]
    buildup_sections: list[tuple[float, float]]
    section_boundaries: list[float]
    tempo_drift: list[tuple[float, float]]
    repeated_rhythmic_motifs: list[tuple[float, float, float]]
    sample_rate: int = 0
    backend: str = "fallback_wav"
    confidence: float = 0.0
    beats: list[BeatFeature] = field(default_factory=list)
    onsets: list[OnsetFeature] = field(default_factory=list)
    downbeat_candidates: list[float] = field(default_factory=list)
    tempo_map: list[tuple[float, float]] = field(default_factory=list)
    tempo_estimate: float = 0.0
    fallback_flux: list[tuple[float, float]] = field(default_factory=list)
    confidence_report: AudioConfidenceReport | None = None


AudioAnalysisResult = AudioFeatures


def load_audio(audio_file: str | Path, *, target_duration: float | None = None) -> AudioBuffer:
    """Load a PCM WAV file into mono float samples.

    The project has no mandatory audio dependency yet. This conservative loader
    keeps phase/timing deterministic for the first implementation stage.
    """

    path = Path(audio_file)
    if not path.exists():
        raise FileNotFoundError(f"audio_file not found: {path}")
    if path.suffix.lower() != ".wav":
        raise ValueError(
            "Only PCM WAV is supported by the built-in analyzer. "
            "Install an external audio frontend later for mp3/ogg/flac."
        )

    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frame_count = reader.getnframes()
        if target_duration is not None:
            frame_count = min(frame_count, int(max(0.0, target_duration) * sample_rate))
        raw = reader.readframes(frame_count)

    samples = _decode_pcm(raw, channels=channels, sample_width=sample_width)
    return AudioBuffer(samples=samples, sample_rate=sample_rate)


def analyze_audio(
    audio_or_file: AudioBuffer | str | Path,
    *,
    song_offset: float = 0.0,
    target_duration: float | None = None,
    frame_size: int = 2048,
    hop_size: int = 512,
    mel_bins: int = 24,
    backend: str = "auto",
) -> AudioFeatures:
    if not isinstance(audio_or_file, AudioBuffer) and backend in {"auto", "librosa"}:
        librosa_result = _try_analyze_with_librosa(
            Path(audio_or_file),
            song_offset=song_offset,
            target_duration=target_duration,
            mel_bins=mel_bins,
            required=backend == "librosa",
        )
        if librosa_result is not None:
            return librosa_result

    if isinstance(audio_or_file, AudioBuffer):
        audio = audio_or_file
    else:
        audio = load_audio(audio_or_file, target_duration=target_duration)

    samples = audio.samples
    sample_rate = audio.sample_rate
    if target_duration is not None:
        samples = samples[: int(max(0.0, target_duration) * sample_rate)]

    frame_features = _extract_frame_features(
        samples,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        mel_bins=mel_bins,
    )
    _postprocess_frame_features(frame_features)

    duration = len(samples) / sample_rate if sample_rate else 0.0
    bpm = _estimate_bpm(frame_features, hop_size=hop_size, sample_rate=sample_rate)
    beat_times = _build_beat_grid(
        frame_features,
        bpm=bpm,
        duration=duration,
        song_offset=song_offset,
    )
    downbeat_times = [time_value for idx, time_value in enumerate(beat_times) if idx % 4 == 0]
    onsets = _detect_onsets(frame_features)
    onset_times = [onset.time for onset in onsets]
    sections = _segment_sections(
        frame_features,
        beat_times=beat_times,
        duration=duration,
    )
    beat_features = _build_beat_features(frame_features, beat_times)
    tempo_map = _estimate_tempo_drift(beat_times)
    confidence_report = _analysis_confidence_report(
        frame_features=frame_features,
        beat_features=beat_features,
        sections=sections,
        tempo_map=tempo_map,
        backend="fallback_wav",
    )
    confidence = confidence_report.overall

    low_energy_sections = [
        (section.start_time, section.end_time)
        for section in sections
        if section.section_type in {"calm", "silence", "intro", "outro"}
    ]
    drop_points = [
        section.start_time for section in sections if section.section_type == "drop"
    ]
    buildup_sections = [
        (section.start_time, section.end_time)
        for section in sections
        if section.section_type == "buildup"
    ]
    section_boundaries = [section.start_time for section in sections[1:]]
    repeated_motifs = _detect_repeated_rhythmic_motifs(
        frame_features,
        beat_times=beat_times,
    )

    return AudioFeatures(
        duration=duration,
        bpm=bpm,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        onset_times=onset_times,
        rms_envelope=[(frame.time, frame.rms) for frame in frame_features],
        onset_envelope=[(frame.time, frame.onset_strength) for frame in frame_features],
        spectral_flux=[(frame.time, frame.spectral_flux) for frame in frame_features],
        mel_spectrogram=[frame.mel_bins for frame in frame_features],
        chroma=_build_chroma(frame_features),
        frame_features=frame_features,
        beat_features=beat_features,
        sections=sections,
        low_energy_sections=low_energy_sections,
        drop_points=drop_points,
        buildup_sections=buildup_sections,
        section_boundaries=section_boundaries,
        tempo_drift=tempo_map,
        repeated_rhythmic_motifs=repeated_motifs,
        sample_rate=sample_rate,
        backend="fallback_wav",
        confidence=confidence,
        beats=beat_features,
        onsets=onsets,
        downbeat_candidates=downbeat_times,
        tempo_map=tempo_map,
        tempo_estimate=bpm,
        fallback_flux=[(frame.time, frame.spectral_flux) for frame in frame_features],
        confidence_report=confidence_report,
    )


def _try_analyze_with_librosa(
    audio_file: Path,
    *,
    song_offset: float,
    target_duration: float | None,
    mel_bins: int,
    required: bool,
) -> AudioFeatures | None:
    try:
        import librosa  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        if required:
            raise RuntimeError("librosa backend requested but librosa is not available") from exc
        return None

    try:
        y, sr = librosa.load(
            str(audio_file),
            sr=None,
            mono=True,
            duration=target_duration,
        )
    except Exception as exc:  # noqa: BLE001
        if required:
            raise
        return None

    duration = float(librosa.get_duration(y=y, sr=sr))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    frame_times = librosa.frames_to_time(range(len(onset_env)), sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
    bpm = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 120.0
    beat_times = [float(t) for t in librosa.frames_to_time(beat_frames, sr=sr)]
    beat_times = [time for time in beat_times if time >= max(0.0, song_offset)]
    downbeat_times = [time for idx, time in enumerate(beat_times) if idx % 4 == 0]

    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(range(len(rms)), sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_bins)
    mel_norm = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_flux = _librosa_flux(mel_norm)

    frame_features: list[FrameFeature] = []
    frame_count = min(len(frame_times), len(onset_env), len(rms), mel_norm.shape[1])
    for idx in range(frame_count):
        mel_col = [float(value) for value in mel_norm[:, idx]]
        peak = max(mel_col, default=0.0)
        floor = min(mel_col, default=0.0)
        denom = max(1e-9, peak - floor)
        mel_scaled = [(value - floor) / denom for value in mel_col]
        frame_features.append(
            FrameFeature(
                time=float(frame_times[idx]),
                rms=float(rms[idx]),
                onset_strength=float(onset_env[idx]),
                spectral_flux=float(spectral_flux[idx]) if idx < len(spectral_flux) else 0.0,
                mel_bins=mel_scaled,
            )
        )
    _postprocess_frame_features(frame_features)

    onset_frames = list(librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env))
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onsets = [
        OnsetFeature(
            time=float(time),
            strength=frame_features[min(int(frame), len(frame_features) - 1)].onset_strength
            if frame_features else 0.0,
        )
        for frame, time in zip(onset_frames, onset_times)
    ]
    _normalize_onset_features(onsets)
    sections = _segment_sections(frame_features, beat_times=beat_times, duration=duration)
    beat_features = _build_beat_features(frame_features, beat_times)
    tempo_map = _estimate_tempo_drift(beat_times)
    confidence_report = _analysis_confidence_report(
        frame_features=frame_features,
        beat_features=beat_features,
        sections=sections,
        tempo_map=tempo_map,
        backend="librosa",
    )
    confidence = confidence_report.overall

    chroma_rows: list[list[float]] = []
    for idx in range(chroma.shape[1]):
        col = [float(value) for value in chroma[:, idx]]
        total = sum(col)
        chroma_rows.append([value / total for value in col] if total > 0 else col)

    return AudioFeatures(
        duration=duration,
        bpm=bpm,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        onset_times=[onset.time for onset in onsets],
        rms_envelope=[(float(t), float(v)) for t, v in zip(rms_times, rms)],
        onset_envelope=[(float(t), float(v)) for t, v in zip(frame_times, onset_env)],
        spectral_flux=[(frame.time, frame.spectral_flux) for frame in frame_features],
        mel_spectrogram=[frame.mel_bins for frame in frame_features],
        chroma=chroma_rows,
        frame_features=frame_features,
        beat_features=beat_features,
        sections=sections,
        low_energy_sections=[
            (section.start_time, section.end_time)
            for section in sections
            if section.section_type in {"break", "intro", "outro"}
        ],
        drop_points=[section.start_time for section in sections if section.section_type == "drop"],
        buildup_sections=[
            (section.start_time, section.end_time)
            for section in sections
            if section.section_type == "buildup"
        ],
        section_boundaries=[section.start_time for section in sections[1:]],
        tempo_drift=tempo_map,
        repeated_rhythmic_motifs=_detect_repeated_rhythmic_motifs(
            frame_features,
            beat_times=beat_times,
        ),
        sample_rate=int(sr),
        backend="librosa",
        confidence=confidence,
        beats=beat_features,
        onsets=onsets,
        downbeat_candidates=downbeat_times,
        tempo_map=tempo_map,
        tempo_estimate=bpm,
        fallback_flux=[(frame.time, frame.spectral_flux) for frame in frame_features],
        confidence_report=confidence_report,
    )


def _librosa_flux(mel_db: Any) -> list[float]:
    values: list[float] = [0.0]
    for idx in range(1, mel_db.shape[1]):
        diff = mel_db[:, idx] - mel_db[:, idx - 1]
        values.append(float(sum(max(0.0, float(value)) for value in diff)))
    peak = max(values, default=0.0)
    return [value / peak for value in values] if peak > 0 else values


def _decode_pcm(raw: bytes, *, channels: int, sample_width: int) -> list[float]:
    if channels < 1:
        raise ValueError("WAV channel count must be >= 1")

    if sample_width == 1:
        values = [((byte - 128) / 128.0) for byte in raw]
    elif sample_width == 2:
        count = len(raw) // 2
        ints = struct.unpack("<" + "h" * count, raw[: count * 2])
        values = [value / 32768.0 for value in ints]
    elif sample_width == 4:
        count = len(raw) // 4
        ints = struct.unpack("<" + "i" * count, raw[: count * 4])
        values = [value / 2147483648.0 for value in ints]
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels == 1:
        return values

    mono: list[float] = []
    for idx in range(0, len(values) - channels + 1, channels):
        mono.append(sum(values[idx : idx + channels]) / channels)
    return mono


def _extract_frame_features(
    samples: list[float],
    *,
    sample_rate: int,
    frame_size: int,
    hop_size: int,
    mel_bins: int,
) -> list[FrameFeature]:
    if not samples:
        return []

    frames: list[FrameFeature] = []
    prev_bands = [0.0] * mel_bins
    prev_rms = 0.0

    for start in range(0, max(1, len(samples) - frame_size + 1), hop_size):
        frame = samples[start : start + frame_size]
        if len(frame) < frame_size:
            frame = frame + [0.0] * (frame_size - len(frame))
        time_value = start / sample_rate
        rms = math.sqrt(sum(value * value for value in frame) / len(frame))
        bands = _coarse_log_band_energies(frame, sample_rate=sample_rate, bins=mel_bins)
        spectral_flux = sum(max(0.0, band - prev) for band, prev in zip(bands, prev_bands))
        onset_strength = max(0.0, rms - prev_rms) + 0.35 * spectral_flux
        frames.append(
            FrameFeature(
                time=time_value,
                rms=rms,
                onset_strength=onset_strength,
                spectral_flux=spectral_flux,
                mel_bins=bands,
            )
        )
        prev_bands = bands
        prev_rms = rms

    if not frames:
        frames.append(
            FrameFeature(
                time=0.0,
                rms=0.0,
                onset_strength=0.0,
                spectral_flux=0.0,
                mel_bins=[0.0] * mel_bins,
            )
        )
    return frames


def _postprocess_frame_features(frame_features: list[FrameFeature]) -> None:
    """Normalize onset/flux and smooth RMS in-place for stable planning."""

    if not frame_features:
        return
    smoothed_rms = _moving_average([frame.rms for frame in frame_features], radius=2)
    onset_values = _normalize_values([frame.onset_strength for frame in frame_features])
    flux_values = _normalize_values([frame.spectral_flux for frame in frame_features])
    for frame, rms, onset, flux in zip(frame_features, smoothed_rms, onset_values, flux_values):
        frame.rms = rms
        frame.onset_strength = onset
        frame.spectral_flux = flux


def _normalize_onset_features(onsets: list[OnsetFeature]) -> None:
    values = _normalize_values([onset.strength for onset in onsets])
    for onset, value in zip(onsets, values):
        onset.strength = value


def _normalize_values(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if abs(high - low) <= 1e-12:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _moving_average(values: list[float], *, radius: int) -> list[float]:
    if not values:
        return []
    result: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        result.append(sum(values[start:end]) / max(1, end - start))
    return result


def _coarse_log_band_energies(
    frame: list[float],
    *,
    sample_rate: int,
    bins: int,
) -> list[float]:
    # A lightweight mel-like descriptor. It is not a replacement for a real STFT
    # frontend, but it preserves coarse spectral change for scoring/planning.
    if not frame or sample_rate <= 0:
        return [0.0] * bins

    max_points = 512
    step = max(1, len(frame) // max_points)
    compact = frame[::step][:max_points]
    n = len(compact)
    if n == 0:
        return [0.0] * bins

    min_freq = 60.0
    max_freq = min(8000.0, sample_rate / 2.0)
    energies: list[float] = []
    for idx in range(bins):
        ratio = idx / max(1, bins - 1)
        freq = min_freq * ((max_freq / min_freq) ** ratio)
        omega = 2.0 * math.pi * freq / sample_rate * step
        real = 0.0
        imag = 0.0
        for sample_idx, sample in enumerate(compact):
            phase = omega * sample_idx
            real += sample * math.cos(phase)
            imag -= sample * math.sin(phase)
        energies.append(math.sqrt(real * real + imag * imag) / n)

    peak = max(energies, default=0.0)
    if peak <= 1e-12:
        return energies
    return [energy / peak for energy in energies]


def _estimate_bpm(
    frame_features: list[FrameFeature],
    *,
    hop_size: int,
    sample_rate: int,
) -> float:
    if len(frame_features) < 4 or sample_rate <= 0:
        return 120.0

    values = [frame.onset_strength for frame in frame_features]
    mean_value = sum(values) / len(values)
    values = [max(0.0, value - mean_value) for value in values]
    hop_seconds = hop_size / sample_rate
    best_bpm = 120.0
    best_score = -1.0

    for bpm in range(60, 201):
        lag = int(round((60.0 / bpm) / hop_seconds))
        if lag <= 0 or lag >= len(values):
            continue
        score = sum(values[idx] * values[idx - lag] for idx in range(lag, len(values)))
        if score > best_score:
            best_score = score
            best_bpm = float(bpm)

    return best_bpm


def _build_beat_grid(
    frame_features: list[FrameFeature],
    *,
    bpm: float,
    duration: float,
    song_offset: float,
) -> list[float]:
    if duration <= 0:
        return []

    beat_interval = 60.0 / max(1.0, bpm)
    onset_values = [frame.onset_strength for frame in frame_features]
    threshold = _mean(onset_values) + _std(onset_values) * 0.5
    first_beat = max(0.0, song_offset)
    for frame in frame_features:
        if frame.time >= first_beat and frame.onset_strength >= threshold:
            first_beat = frame.time
            break

    beat_times: list[float] = []
    current = first_beat
    while current <= duration + 1e-6:
        beat_times.append(round(current, 6))
        current += beat_interval
    return beat_times


def _detect_onsets(frame_features: list[FrameFeature]) -> list[OnsetFeature]:
    values = [frame.onset_strength for frame in frame_features]
    threshold = _mean(values) + _std(values)
    onsets = [
        OnsetFeature(time=frame.time, strength=frame.onset_strength)
        for frame in frame_features
        if frame.onset_strength >= threshold and frame.onset_strength > 0.0
    ]
    _normalize_onset_features(onsets)
    return onsets


def _build_beat_features(
    frame_features: list[FrameFeature],
    beat_times: list[float],
) -> list[BeatFeature]:
    result: list[BeatFeature] = []
    for idx, beat_time in enumerate(beat_times):
        window = _frames_in_window(frame_features, beat_time - 0.12, beat_time + 0.12)
        strength = max((frame.onset_strength for frame in window), default=0.0)
        local_energy = _mean([frame.rms for frame in window])
        rhythmic_density = _density_around(frame_features, beat_time)
        result.append(
            BeatFeature(
                index=idx,
                time=beat_time,
                strength=strength,
                is_downbeat=(idx % 4 == 0),
                local_energy=local_energy,
                rhythmic_density=rhythmic_density,
            )
        )
    return result


def _segment_sections(
    frame_features: list[FrameFeature],
    *,
    beat_times: list[float],
    duration: float,
) -> list[SectionFeature]:
    if duration <= 0:
        return []
    if not beat_times:
        beat_times = [0.0, duration]

    section_edges = [0.0]
    step = max(4, min(16, len(beat_times)))
    for idx in range(step, len(beat_times), step):
        section_edges.append(beat_times[idx])
    section_edges.extend(_flux_section_boundaries(frame_features, duration=duration))
    section_edges = _merge_section_edges(section_edges, duration=duration)
    if section_edges[-1] < duration:
        section_edges.append(duration)

    all_energy = [frame.rms for frame in frame_features]
    median_energy = statistics.median(all_energy) if all_energy else 0.0
    max_energy = max(all_energy, default=1.0)
    sections: list[SectionFeature] = []
    previous_energy = 0.0

    for idx, (start, end) in enumerate(zip(section_edges, section_edges[1:])):
        frames = _frames_in_window(frame_features, start, end)
        mean_energy = _mean([frame.rms for frame in frames])
        density = _mean([frame.onset_strength for frame in frames])
        trend = mean_energy - previous_energy
        normalized = mean_energy / max(max_energy, 1e-9)
        energy_peak = max([frame.rms for frame in frames], default=0.0)
        if normalized < 0.12:
            section_type = "break"
        elif idx == 0 and mean_energy <= median_energy:
            section_type = "intro"
        elif idx == len(section_edges) - 2 and mean_energy <= median_energy:
            section_type = "outro"
        elif trend > median_energy * 0.25:
            section_type = "buildup"
        elif mean_energy >= median_energy * 1.25 and density > 0:
            section_type = "drop"
        elif mean_energy <= median_energy * 0.8:
            section_type = "break"
        else:
            section_type = "normal"

        sections.append(
            SectionFeature(
                section_id=idx,
                start_time=start,
                end_time=end,
                section_type=section_type,
                mean_energy=mean_energy,
                rhythmic_density=density,
                confidence=min(1.0, 0.5 + abs(trend) / max(max_energy, 1e-9)),
                energy_peak=energy_peak,
            )
        )
        previous_energy = mean_energy

    return sections


def _flux_section_boundaries(
    frame_features: list[FrameFeature],
    *,
    duration: float,
) -> list[float]:
    if duration <= 0 or len(frame_features) < 4:
        return []
    flux_values = [frame.spectral_flux for frame in frame_features]
    energy_values = [frame.rms for frame in frame_features]
    flux_threshold = _mean(flux_values) + _std(flux_values) * 1.25
    energy_threshold = _std(energy_values) * 0.65
    candidates: list[float] = []
    previous_energy = energy_values[0]
    for frame, energy in zip(frame_features[1:], energy_values[1:]):
        energy_delta = abs(energy - previous_energy)
        if frame.spectral_flux >= flux_threshold or energy_delta >= energy_threshold:
            if 0.35 < frame.time < duration - 0.35:
                candidates.append(frame.time)
        previous_energy = energy
    return candidates


def _merge_section_edges(edges: list[float], *, duration: float) -> list[float]:
    min_gap = max(0.75, duration / 12.0)
    merged: list[float] = []
    for edge in sorted(set(round(max(0.0, min(duration, edge)), 6) for edge in edges)):
        if not merged or edge - merged[-1] >= min_gap:
            merged.append(edge)
    if not merged or merged[0] != 0.0:
        merged.insert(0, 0.0)
    if merged[-1] < duration:
        merged.append(duration)
    return merged


def _estimate_tempo_drift(beat_times: list[float]) -> list[tuple[float, float]]:
    drift: list[tuple[float, float]] = []
    if len(beat_times) < 5:
        return drift
    for idx in range(0, len(beat_times) - 4, 4):
        span = beat_times[idx + 4] - beat_times[idx]
        if span > 0:
            drift.append((beat_times[idx], 240.0 / span))
    return drift


def _detect_repeated_rhythmic_motifs(
    frame_features: list[FrameFeature],
    *,
    beat_times: list[float],
) -> list[tuple[float, float, float]]:
    if len(beat_times) < 8:
        return []

    patterns: dict[tuple[int, ...], list[tuple[float, float]]] = {}
    for idx in range(0, len(beat_times) - 4, 4):
        start = beat_times[idx]
        end = beat_times[idx + 4]
        frames = _frames_in_window(frame_features, start, end)
        if not frames:
            continue
        threshold = _mean([frame.onset_strength for frame in frames])
        pattern = tuple(1 if frame.onset_strength >= threshold else 0 for frame in frames[:16])
        patterns.setdefault(pattern, []).append((start, end))

    motifs: list[tuple[float, float, float]] = []
    for occurrences in patterns.values():
        if len(occurrences) < 2:
            continue
        confidence = min(1.0, len(occurrences) / 4.0)
        for start, end in occurrences[:4]:
            motifs.append((start, end, confidence))
    return motifs


def _build_chroma(frame_features: list[FrameFeature]) -> list[list[float]]:
    chroma_rows: list[list[float]] = []
    for frame in frame_features:
        row = [0.0] * 12
        for idx, energy in enumerate(frame.mel_bins):
            row[idx % 12] += energy
        total = sum(row)
        if total > 0:
            row = [value / total for value in row]
        chroma_rows.append(row)
    return chroma_rows


def _frames_in_window(
    frame_features: list[FrameFeature],
    start_time: float,
    end_time: float,
) -> list[FrameFeature]:
    return [frame for frame in frame_features if start_time <= frame.time < end_time]


def _density_around(frame_features: list[FrameFeature], time_value: float) -> float:
    window = _frames_in_window(frame_features, time_value - 0.25, time_value + 0.25)
    if not window:
        return 0.0
    values = [frame.onset_strength for frame in window]
    threshold = _mean(values)
    return sum(1 for value in values if value >= threshold and value > 0) / len(values)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def _analysis_confidence(
    *,
    frame_features: list[FrameFeature],
    beat_features: list[BeatFeature],
    sections: list[SectionFeature],
) -> float:
    return _analysis_confidence_report(
        frame_features=frame_features,
        beat_features=beat_features,
        sections=sections,
        tempo_map=[],
        backend="unknown",
    ).overall


def _analysis_confidence_report(
    *,
    frame_features: list[FrameFeature],
    beat_features: list[BeatFeature],
    sections: list[SectionFeature],
    tempo_map: list[tuple[float, float]],
    backend: str,
) -> AudioConfidenceReport:
    if not frame_features:
        return AudioConfidenceReport(
            overall=0.0,
            bpm_confidence=0.0,
            beat_confidence=0.0,
            onset_confidence=0.0,
            section_confidence=0.0,
            tempo_stability=0.0,
            backend=backend,
            warnings=["empty_audio_features"],
        )
    onset_values = [frame.onset_strength for frame in frame_features]
    rms_values = [frame.rms for frame in frame_features]
    onset_contrast = _std(onset_values) / max(_mean(onset_values), 1e-9)
    energy_contrast = _std(rms_values) / max(_mean(rms_values), 1e-9)
    onset_confidence = min(1.0, onset_contrast)
    bpm_confidence = 0.35 + 0.65 * min(1.0, len(beat_features) / max(4.0, len(frame_features) / 16.0))
    beat_term = min(1.0, len(beat_features) / 16.0)
    section_term = min(1.0, len(sections) / 4.0)
    if len(tempo_map) >= 2:
        tempos = [tempo for _, tempo in tempo_map]
        tempo_stability = max(0.0, 1.0 - (_std(tempos) / max(_mean(tempos), 1e-9)))
    else:
        tempo_stability = 0.65 if len(beat_features) >= 4 else 0.35
    overall = max(
        0.05,
        min(
            1.0,
            0.20 * bpm_confidence
            + 0.20 * min(1.0, onset_contrast)
            + 0.25 * min(1.0, energy_contrast)
            + 0.20 * beat_term
            + 0.20 * section_term
            + 0.15 * tempo_stability,
        ),
    )
    warnings: list[str] = []
    if overall < 0.35:
        warnings.append("low_audio_analysis_confidence")
    if len(beat_features) < 4:
        warnings.append("few_detected_beats")
    if onset_confidence < 0.2:
        warnings.append("weak_onset_contrast")
    return AudioConfidenceReport(
        overall=overall,
        bpm_confidence=max(0.0, min(1.0, bpm_confidence)),
        beat_confidence=beat_term,
        onset_confidence=max(0.0, min(1.0, onset_confidence)),
        section_confidence=section_term,
        tempo_stability=max(0.0, min(1.0, tempo_stability)),
        backend=backend,
        warnings=warnings,
    )
