"""Synthesize Azerbaijani test sentences from a StyleTTS2 checkpoint (adapted from Demo/Inference_LibriTTS.ipynb)."""
import argparse
import csv
import json
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
ROOT = Path(__file__).resolve().parents[1]
STTS = ROOT / "StyleTTS2"
sys.path.insert(0, str(STTS))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
import yaml  # noqa: E402

from az_text import phonemize_texts  # noqa: E402

DEFAULT_TEXTS = [
    "Salam! Bu gün hava çox gözəldir, gəlin parka gedək.",
    "Siz sabah bizim ofisə gələ biləcəksinizmi?",
    "Müraciətiniz qeydə alındı, tezliklə əməkdaşlarımız sizinlə əlaqə saxlayacaq və bütün suallarınızı cavablandıracaqlar.",
]
to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)


INFERENCE_DEFAULTS = {"alpha": 0.3, "beta": 0.7, "embedding_scale": 1.5, "steps": 10, "speed": 1.0, "ref": None}


def load_inference_defaults(path=ROOT / "configs/inference.json"):
    """Built-in defaults, overridden by the tuned settings eval_tts.py writes (if present)."""
    d = dict(INFERENCE_DEFAULTS)
    if Path(path).exists():
        d.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return d


def g2p_for_config(config):
    """Runs trained before the rule-based G2P have no `g2p` key: their labels came from espeak."""
    return config.get("g2p", "espeak")


def default_config(checkpoint):
    """The run config saved next to the checkpoint (it records the run's g2p and sigma_data), else the
    original run's, else upstream LibriTTS."""
    for cfg in (Path(checkpoint).parent / "config_ft_az.yml", ROOT / "runs/az_ft/config_ft_az.yml"):
        if cfg.exists():
            return cfg
    return STTS / "Models/LibriTTS/config.yml"


def tuned_for_other_run(defaults, checkpoint):
    """True if configs/inference.json was tuned on a checkpoint from a different run directory."""
    tuned = defaults.get("checkpoint")
    return bool(tuned) and (ROOT / tuned).resolve().parent != Path(checkpoint).resolve().parent


def pick_device(device=None):
    """`device` if given, else the GPU when torch can use one, else the CPU."""
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def check_weights(checkpoint):
    """Fail with a usable message when the checkpoint is missing or is still a Git LFS pointer stub."""
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"TTS checkpoint not found: {checkpoint}")
    if checkpoint.stat().st_size < 2**20:
        raise RuntimeError(f"{checkpoint} is a Git LFS pointer, not the model: run `git lfs install` and "
                           f"`git lfs pull` in the repo")


def default_out_dir(checkpoint):
    """<run dir>/samples/<checkpoint stem>: runs share checkpoint names, so keep samples per run."""
    return Path(checkpoint).parent / "samples" / Path(checkpoint).stem


def metadata_text(wav_name):
    """Original text of a dataset clip (data/wavs/000123.wav <- metadata.csv 000123.mp3)."""
    stem = Path(wav_name).stem
    with open(ROOT / "metadata.csv", encoding="utf-8", newline="") as f:
        return next(r["text"] for r in csv.DictReader(f) if Path(r["filename"]).stem == stem)


def predict_durations(dur_logits, speed=1.0):
    """Frames per token from the duration projection's logits; speed > 1 speaks faster."""
    duration = torch.sigmoid(dur_logits).sum(axis=-1) / speed
    return torch.round(duration.squeeze()).clamp(min=1)


def average_styles(styles):
    return torch.stack(styles).mean(dim=0)


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def preprocess(wave):
    mel = to_mel(torch.from_numpy(wave).float())
    return (torch.log(1e-5 + mel.unsqueeze(0)) + 4) / 4


class Synth:
    def __init__(self, config_path, checkpoint, device=None):
        check_weights(checkpoint)
        device = pick_device(device)
        cwd = os.getcwd()
        os.chdir(STTS)  # upstream config paths (Utils/...) are relative to the repo
        try:
            from models import build_model, load_ASR_models, load_F0_models
            from utils import recursive_munch
            from Utils.PLBERT.util import load_plbert
            from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
            from text_utils import TextCleaner
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
            pitch_extractor = load_F0_models(config["F0_path"])
            plbert = load_plbert(config["PLBERT_dir"])
            self.params = recursive_munch(config["model_params"])
            self.model = build_model(self.params, text_aligner, pitch_extractor, plbert)
            self.g2p = g2p_for_config(config)
            print(f"[synthesize] config {config_path} (g2p: {self.g2p}, device: {device})")
        finally:
            os.chdir(cwd)
        state = torch.load(checkpoint, map_location="cpu")["net"]
        for key in self.model:
            if key in state:
                sd = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state[key].items())
                result = self.model[key].load_state_dict(sd, strict=False)
                missing, unexpected = result.missing_keys, result.unexpected_keys
                if missing or unexpected:
                    print(f"[synthesize] WARNING {key}: {len(missing)} missing / {len(unexpected)} unexpected "
                          f"state_dict key(s) after loading checkpoint")
                    if missing:
                        print(f"  missing: {missing}")
                    if unexpected:
                        print(f"  unexpected: {unexpected}")
        self.device = device
        for key in self.model:
            self.model[key].eval().to(device)
        self.sampler = DiffusionSampler(self.model.diffusion.diffusion, sampler=ADPM2Sampler(),
                                        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
                                        clamp=False)
        self.cleaner = TextCleaner()

    @torch.no_grad()
    def compute_style(self, path):
        wave, _ = librosa.load(path, sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        mel = preprocess(audio).to(self.device)
        return torch.cat([self.model.style_encoder(mel.unsqueeze(1)),
                          self.model.predictor_encoder(mel.unsqueeze(1))], dim=1)

    @torch.no_grad()
    def compute_styles(self, paths):
        """Mean style over several reference clips (smooths out one clip's idiosyncratic prosody)."""
        return average_styles([self.compute_style(p) for p in paths])

    @torch.no_grad()
    def __call__(self, ipa, ref_s, alpha=0.3, beta=0.7, steps=10, embedding_scale=1.5, speed=1.0):
        m, dev = self.model, self.device
        tokens = self.cleaner(ipa)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(dev).unsqueeze(0)
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(dev)
        text_mask = length_to_mask(input_lengths).to(dev)
        t_en = m.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = m.bert(tokens, attention_mask=(~text_mask).int())
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
        s_pred = self.sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(dev), embedding=bert_dur,
                              embedding_scale=embedding_scale, features=ref_s, num_steps=steps).squeeze(1)
        s, ref = s_pred[:, 128:], s_pred[:, :128]
        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]
        d = m.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = m.predictor.lstm(d)
        pred_dur = predict_durations(m.predictor.duration_proj(x), speed)
        aln = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c = 0
        for i in range(aln.size(0)):
            aln[i, c:c + int(pred_dur[i].data)] = 1
            c += int(pred_dur[i].data)
        aln = aln.unsqueeze(0).to(dev)
        en = d.transpose(-1, -2) @ aln
        asr = t_en @ aln
        if self.params.decoder.type == "hifigan":
            en = torch.cat([en[:, :, :1], en[:, :, :-1]], dim=-1)
            asr = torch.cat([asr[:, :, :1], asr[:, :, :-1]], dim=-1)
        F0_pred, N_pred = m.predictor.F0Ntrain(en, s)
        out = m.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        return out.squeeze().cpu().numpy()[..., :-50]


def main():
    d = load_inference_defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="default: the one named in configs/inference.json")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ref", action="append", default=None,
                    help="style reference wav; repeat to average several (default: configs/inference.json, "
                         "else the first val clip)")
    ap.add_argument("--text", action="append", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--alpha", type=float, default=d["alpha"])
    ap.add_argument("--beta", type=float, default=d["beta"])
    ap.add_argument("--steps", type=int, default=d["steps"])
    ap.add_argument("--embedding_scale", type=float, default=d["embedding_scale"])
    ap.add_argument("--speed", type=float, default=d["speed"], help=">1 speaks faster")
    a = ap.parse_args()

    # Resolve any user-supplied relative paths against the caller's cwd now --
    # Synth.__init__ chdir's into StyleTTS2/ internally, so a relative path
    # captured only as a string would otherwise be (re)interpreted relative to
    # StyleTTS2/ instead of where the user invoked this script from.
    if not (a.checkpoint or d.get("checkpoint")):
        ap.error("--checkpoint is required (configs/inference.json names none)")
    checkpoint = Path(a.checkpoint).resolve() if a.checkpoint else (ROOT / d["checkpoint"]).resolve()
    config = Path(a.config).resolve() if a.config else default_config(checkpoint)
    val_first = (ROOT / "data/val_list.txt").read_text(encoding="utf-8").splitlines()[0].split("|")
    have_gt = (ROOT / "data/wavs" / val_first[0]).exists()  # the dataset audio is not in a fresh clone
    if a.ref:
        refs = [Path(r).resolve() for r in a.ref]
    elif d["ref"]:
        refs = [ROOT / r for r in d["ref"]]  # inference.json paths are relative to the repo root
    else:
        refs = [ROOT / "data/wavs" / val_first[0]]
    out_dir = Path(a.out_dir).resolve() if a.out_dir else default_out_dir(checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    if tuned_for_other_run(d, checkpoint):
        print(f"[synthesize] note: configs/inference.json was tuned on {d['checkpoint']}; "
              f"re-run eval_tts.py --tune for this run's checkpoints")
    synth = Synth(config, checkpoint, a.device)
    texts = a.text or DEFAULT_TEXTS
    if a.text is None and have_gt:  # add the val item for A/B against the real recording
        texts = texts + [metadata_text(val_first[0])]
        shutil.copy(ROOT / "data/wavs" / val_first[0], out_dir / f"gt_{val_first[0]}")
    ipas = phonemize_texts(texts, g2p=synth.g2p)  # the G2P this checkpoint was trained with

    ref_s = synth.compute_styles(refs)
    for n, (t, ipa) in enumerate(zip(texts, ipas), 1):
        wav = synth(ipa, ref_s, a.alpha, a.beta, a.steps, a.embedding_scale, a.speed)
        path = out_dir / f"{n:02d}.wav"
        sf.write(path, wav, 24000)
        print(f"{path}  ({len(wav)/24000:.1f}s)  {t}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # Azerbaijani letters on a cp1252 Windows console
    main()
