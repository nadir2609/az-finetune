// Page of the voice assistant: microphone -> WebSocket -> server, reply audio -> speaker.
const OUT_RATE = 24000;   // the server's reply audio
const TAIL_S = 0.3;       // keep the microphone muted this long after the reply ends
const CLOSED = {
  4401: "Giriş kodu yanlışdır.",
  4429: "Demo hazırda doludur. Bir az sonra yenidən cəhd edin.",
  4408: "Uzun müddət səs gəlmədiyi üçün bağlantı bağlandı.",
};
const $ = (id) => document.getElementById(id);

let socket = null, stream = null, micCtx = null, playCtx = null;
let playUntil = 0;        // playCtx time at which the queued reply audio ends
let thinking = false;

function setStatus(text) { $("status").textContent = text; }

function addLine(who, text) {
  const p = document.createElement("p"), label = document.createElement("span");
  label.className = "who";
  label.textContent = who + ":";
  p.append(label, text);
  $("log").append(p);
  p.scrollIntoView({ block: "end" });
}

// Half-duplex: while the reply plays (and shortly after), nothing from the microphone is sent,
// so the assistant never hears its own voice from the speaker.
function speaking() { return playCtx.currentTime < playUntil + TAIL_S; }

function play(data) {
  const pcm = new Int16Array(data);
  if (!pcm.length) return;
  const buffer = playCtx.createBuffer(1, pcm.length, OUT_RATE), channel = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
  const source = playCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(playCtx.destination);
  playUntil = Math.max(playUntil, playCtx.currentTime + 0.05);
  source.start(playUntil);
  playUntil += buffer.duration;
  thinking = false;
}

function stop(message) {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  if (micCtx) micCtx.close();
  if (playCtx) playCtx.close();
  stream = micCtx = playCtx = socket = null;
  $("start").disabled = false;
  setStatus(message);
}

async function start() {
  $("start").disabled = true;
  playCtx = new AudioContext();  // created in the tap handler: phones only allow audio after a tap
  playCtx.resume();
  try {
    stream = await navigator.mediaDevices.getUserMedia(
      { audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
  } catch (e) {
    return stop("Mikrofona icazə verilmədi. Brauzerin tənzimləmələrində icazə verin.");
  }
  try {
    micCtx = new AudioContext({ sampleRate: 16000 });  // let the browser resample where it can
  } catch (e) {
    micCtx = new AudioContext();
  }
  await micCtx.audioWorklet.addModule("/static/mic-worklet.js");
  const mic = new AudioWorkletNode(micCtx, "mic");
  micCtx.createMediaStreamSource(stream).connect(mic);

  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${scheme}//${location.host}/ws?code=${encodeURIComponent($("code").value)}`);
  socket.binaryType = "arraybuffer";
  mic.port.onmessage = (e) => {
    if (socket && socket.readyState === WebSocket.OPEN && !speaking()) socket.send(e.data);
  };
  socket.onopen = () => setStatus("Dinləyirəm…");
  socket.onmessage = (e) => {
    if (typeof e.data !== "string") return play(e.data);
    const message = JSON.parse(e.data);
    if (message.type === "user_text") {
      addLine("Siz", message.text);
      thinking = true;
    } else if (message.type === "reply_text") {
      addLine("Köməkçi", message.text);
    } else if (message.type === "latency" && message.first_audio !== undefined) {
      $("latency").textContent = `Cavab ${(message.first_audio / 1000).toFixed(1)} saniyəyə başladı`;
    } else if (message.type === "reply_end") {
      thinking = false;
    }
  };
  socket.onclose = (e) => stop(CLOSED[e.code] || "Bağlantı kəsildi. Yenidən başlayın.");
}

setInterval(() => {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  setStatus(playCtx.currentTime < playUntil ? "Danışıram…" : thinking ? "Düşünürəm…" : "Dinləyirəm…");
}, 100);

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  start();
});
