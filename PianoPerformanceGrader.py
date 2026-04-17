import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np


PIANO_MIN_HZ = 27.5
PIANO_MAX_HZ = 4186.01
N_FFT = 2048
HOP_LENGTH = 512
MIN_ONSET_GAP_FRAMES = 10
TEMPO_TOLERANCE_FRAMES = 5


@dataclass
class AnalysisResult:
    note_accuracy: float
    tempo_accuracy: float
    overall_accuracy: float
    ideal_notes: list[str]
    test_notes: list[str]
    ideal_tempo: list[int]
    test_tempo: list[int]


def levenshtein_distance(a: Sequence, b: Sequence) -> int:
    """Return edit distance between two sequences."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[n][m]


def similarity_percent(a: Sequence, b: Sequence) -> float:
    """Convert edit distance into a similarity percentage."""
    if not a and not b:
        return 100.0

    longest = max(len(a), len(b))
    if longest == 0:
        return 100.0

    distance = levenshtein_distance(a, b)
    similarity = (1 - distance / longest) * 100
    return max(0.0, similarity)


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file in mono while keeping its original sample rate."""
    return librosa.load(path, sr=None, mono=True)


def extract_dominant_frequencies(y: np.ndarray, sr: int) -> np.ndarray:
    """Return the dominant frequency at each STFT frame."""
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    frequency_bins = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    dominant_indices = np.argmax(stft, axis=0)
    return frequency_bins[dominant_indices]


def filter_onsets(onset_frames: np.ndarray, min_gap: int = MIN_ONSET_GAP_FRAMES) -> np.ndarray:
    """Remove onsets that are too close together."""
    if len(onset_frames) == 0:
        return np.array([], dtype=int)

    filtered = [int(onset_frames[0])]
    for frame in onset_frames[1:]:
        frame = int(frame)
        if frame - filtered[-1] > min_gap:
            filtered.append(frame)

    return np.array(filtered, dtype=int)


def detect_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """Detect note onset frames."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=HOP_LENGTH
    )
    return filter_onsets(onset_frames)


def hz_to_note_name(freq: float) -> str:
    """Convert a frequency in Hz to a note name like C4 or A#3."""
    if freq <= 0:
        raise ValueError("Frequency must be positive.")

    midi_note = int(round(librosa.hz_to_midi(freq)))
    return librosa.midi_to_note(midi_note, octave=True)


def extract_note_sequence(path: str) -> list[str]:
    """
    Build a note sequence by:
    1) finding onsets
    2) getting dominant frequencies
    3) converting valid frequencies near each onset into note names
    """
    y, sr = load_audio(path)
    dominant_freqs = extract_dominant_frequencies(y, sr)
    onset_frames = detect_onsets(y, sr)

    notes: list[str] = []

    for frame in onset_frames:
        if frame >= len(dominant_freqs):
            continue

        start = max(0, frame - 2)
        end = min(len(dominant_freqs), frame + 3)
        window = dominant_freqs[start:end]

        valid_freqs = window[(window >= PIANO_MIN_HZ) & (window <= PIANO_MAX_HZ)]
        if valid_freqs.size == 0:
            continue

        representative_freq = float(np.median(valid_freqs))
        notes.append(hz_to_note_name(representative_freq))

    return notes


def extract_tempo_intervals(path: str) -> list[int]:
    """Return frame distances between detected onsets."""
    y, sr = load_audio(path)
    onset_frames = detect_onsets(y, sr)

    if len(onset_frames) < 2:
        return []

    return np.diff(onset_frames).astype(int).tolist()


def apply_tempo_tolerance(reference: list[int], candidate: list[int], tolerance: int) -> list[int]:
    """
    Snap candidate tempo intervals to the reference when they are close enough.
    This gives small timing mistakes partial credit.
    """
    adjusted = list(candidate)
    for i in range(min(len(reference), len(candidate))):
        if abs(reference[i] - candidate[i]) <= tolerance:
            adjusted[i] = reference[i]
    return adjusted


def analyze_performance(ideal_path: str, test_path: str) -> AnalysisResult:
    """Run note and tempo analysis between two recordings."""
    ideal_notes = extract_note_sequence(ideal_path)
    test_notes = extract_note_sequence(test_path)

    ideal_tempo = extract_tempo_intervals(ideal_path)
    test_tempo_raw = extract_tempo_intervals(test_path)
    test_tempo = apply_tempo_tolerance(
        ideal_tempo,
        test_tempo_raw,
        TEMPO_TOLERANCE_FRAMES
    )

    note_accuracy = similarity_percent(ideal_notes, test_notes)
    tempo_accuracy = similarity_percent(ideal_tempo, test_tempo)
    overall_accuracy = (note_accuracy + tempo_accuracy) / 2

    return AnalysisResult(
        note_accuracy=note_accuracy,
        tempo_accuracy=tempo_accuracy,
        overall_accuracy=overall_accuracy,
        ideal_notes=ideal_notes,
        test_notes=test_notes,
        ideal_tempo=ideal_tempo,
        test_tempo=test_tempo,
    )


class AudioAnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Performance Audio Analyzer")
        self.geometry("700x540")
        self.resizable(False, False)

        self.ideal_path = tk.StringVar()
        self.test_path = tk.StringVar()
        self.status_text = tk.StringVar(value="Select two audio files to begin.")

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=20)
        main.pack(fill="both", expand=True)

        title = ttk.Label(
            main,
            text="Performance Audio Analyzer",
            font=("Helvetica", 18, "bold")
        )
        title.pack(pady=(0, 10))

        description = ttk.Label(
            main,
            text="Compare a reference recording and a test recording using note and tempo similarity.",
            wraplength=600,
            justify="center"
        )
        description.pack(pady=(0, 20))

        self._build_file_picker(
            parent=main,
            label_text="Reference Audio File",
            path_var=self.ideal_path,
            command=self.select_ideal_file
        )

        self._build_file_picker(
            parent=main,
            label_text="Test Audio File",
            path_var=self.test_path,
            command=self.select_test_file
        )

        ttk.Button(main, text="Analyze Performance", command=self.run_analysis).pack(pady=18)

        ttk.Label(main, textvariable=self.status_text).pack(pady=(0, 14))

        results_frame = ttk.LabelFrame(main, text="Results", padding=16)
        results_frame.pack(fill="x", pady=8)

        self.note_label = ttk.Label(results_frame, text="Note Accuracy: --", font=("Helvetica", 12))
        self.note_label.pack(anchor="w", pady=3)

        self.tempo_label = ttk.Label(results_frame, text="Tempo Accuracy: --", font=("Helvetica", 12))
        self.tempo_label.pack(anchor="w", pady=3)

        self.total_label = ttk.Label(results_frame, text="Overall Accuracy: --", font=("Helvetica", 12, "bold"))
        self.total_label.pack(anchor="w", pady=3)

        details_frame = ttk.LabelFrame(main, text="Detected Sequences", padding=16)
        details_frame.pack(fill="both", expand=True, pady=10)

        self.details_text = tk.Text(details_frame, height=10, wrap="word")
        self.details_text.pack(fill="both", expand=True)
        self.details_text.insert("1.0", "Analysis details will appear here.")
        self.details_text.config(state="disabled")

    def _build_file_picker(
        self,
        parent: ttk.Frame,
        label_text: str,
        path_var: tk.StringVar,
        command
    ) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=8)

        ttk.Label(frame, text=label_text).pack(anchor="w")
        entry = ttk.Entry(frame, textvariable=path_var, width=65)
        entry.pack(side="left", padx=(0, 8), pady=6)
        ttk.Button(frame, text="Browse", command=command).pack(side="left")

    def select_ideal_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Reference Audio File",
            filetypes=[("Audio Files", "*.mp3 *.wav"), ("All Files", "*.*")]
        )
        if path:
            self.ideal_path.set(path)

    def select_test_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Test Audio File",
            filetypes=[("Audio Files", "*.mp3 *.wav"), ("All Files", "*.*")]
        )
        if path:
            self.test_path.set(path)

    def update_details_box(self, result: AnalysisResult) -> None:
        details = (
            f"Ideal Notes: {result.ideal_notes}\n\n"
            f"Test Notes: {result.test_notes}\n\n"
            f"Ideal Tempo Intervals: {result.ideal_tempo}\n\n"
            f"Test Tempo Intervals: {result.test_tempo}"
        )

        self.details_text.config(state="normal")
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert("1.0", details)
        self.details_text.config(state="disabled")

    def run_analysis(self) -> None:
        ideal = self.ideal_path.get().strip()
        test = self.test_path.get().strip()

        if not ideal or not test:
            messagebox.showwarning("Missing File", "Please select both audio files first.")
            return

        if not Path(ideal).exists() or not Path(test).exists():
            messagebox.showerror("File Error", "One or both selected files could not be found.")
            return

        try:
            self.status_text.set("Analyzing audio...")
            self.update_idletasks()

            result = analyze_performance(ideal, test)

            self.note_label.config(text=f"Note Accuracy: {result.note_accuracy:.2f}%")
            self.tempo_label.config(text=f"Tempo Accuracy: {result.tempo_accuracy:.2f}%")
            self.total_label.config(text=f"Overall Accuracy: {result.overall_accuracy:.2f}%")

            self.update_details_box(result)

            self.status_text.set("Analysis complete.")
            self.update_idletasks()

            messagebox.showinfo(
                "Analysis Results",
                f"Note Accuracy: {result.note_accuracy:.2f}%\n"
                f"Tempo Accuracy: {result.tempo_accuracy:.2f}%\n"
                f"Overall Accuracy: {result.overall_accuracy:.2f}%"
            )

        except Exception as exc:
            self.status_text.set("Analysis failed.")
            messagebox.showerror("Analysis Error", f"Something went wrong:\n\n{exc}")


if __name__ == "__main__":
    app = AudioAnalyzerApp()
    app.mainloop()
