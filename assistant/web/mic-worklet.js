// Brings the microphone down to 16 kHz (averaging the input samples that fall into each output sample)
// and posts 512-sample blocks of 16-bit audio, the size the server's speech detector expects.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000; // input samples per output sample (1 if the context is already 16 kHz)
    this.pos = 0;
    this.sum = 0;
    this.count = 0;
    this.block = new Int16Array(512);
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;
    for (let i = 0; i < input.length; i++) {
      this.sum += input[i];
      this.count += 1;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const sample = Math.max(-1, Math.min(1, this.sum / this.count));
        this.sum = 0;
        this.count = 0;
        this.block[this.filled++] = sample * 32767;
        if (this.filled === this.block.length) {
          this.port.postMessage(this.block.buffer.slice(0));
          this.filled = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("mic", MicProcessor);
