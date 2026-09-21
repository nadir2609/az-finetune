"""Synthesize Azerbaijani test sentences from a StyleTTS2 checkpoint (adapted from Demo/Inference_LibriTTS.ipynb)."""
import argparse
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


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def preprocess(wave):
    mel = to_mel(torch.from_numpy(wave).float())
    return (torch.log(1e-5 + mel.unsqueeze(0)) + 4) / 4


class Synth:
    def __init__(self, config_path, checkpoint, device="cuda"):
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
        finally:
            os.chdir(cwd)
        state = torch.load(checkpoint, map_location="cpu")["net"]
        for key in self.model:
            if key in state:
                sd = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state[key].items())
                self.model[key].load_state_dict(sd, strict=False)
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
    def __call__(self, ipa, ref_s, alpha=0.3, beta=0.7, steps=10, embedding_scale=1.5):
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
        duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--text", action="append", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--embedding_scale", type=float, default=1.5)
    a = ap.parse_args()

    run_cfg = ROOT / "runs/az_ft/config_ft_az.yml"
    config = a.config or str(run_cfg if run_cfg.exists() else STTS / "Models/LibriTTS/config.yml")
    val_first = (ROOT / "data/val_list.txt").read_text(encoding="utf-8").splitlines()[0].split("|")
    ref = a.ref or str(ROOT / "data/wavs" / val_first[0])
    out_dir = Path(a.out_dir or ROOT / "runs/az_ft/samples" / Path(a.checkpoint).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = a.text or DEFAULT_TEXTS
    ipas = phonemize_texts(texts)
    if a.text is None:  # add the val item (ground-truth IPA) for A/B against the real recording
        texts, ipas = texts + ["<val> " + val_first[0]], ipas + [val_first[1]]
        shutil.copy(ROOT / "data/wavs" / val_first[0], out_dir / f"gt_{val_first[0]}")

    synth = Synth(config, a.checkpoint)
    ref_s = synth.compute_style(ref)
    for n, (t, ipa) in enumerate(zip(texts, ipas), 1):
        wav = synth(ipa, ref_s, a.alpha, a.beta, a.steps, a.embedding_scale)
        path = out_dir / f"{n:02d}.wav"
        sf.write(path, wav, 24000)
        print(f"{path}  ({len(wav)/24000:.1f}s)  {t}")


if __name__ == "__main__":
    main()
