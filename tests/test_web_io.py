import numpy as np
import pytest

from assistant.web_io import Disconnected, IdleTimeout, SocketMic, SocketPlayer


def pcm(samples):
    return np.asarray(samples, dtype="<i2").tobytes()


def test_one_block_of_16_bit_audio_becomes_512_float_samples():
    mic = SocketMic()
    mic.push(pcm([16384] * 512))
    block = mic.read(timeout=0)
    assert block.dtype == np.float32 and block.shape == (512,)
    assert block[0] == pytest.approx(0.5)
    assert mic.read(timeout=0) is None


def test_frames_of_any_size_are_reblocked_without_losing_alignment():
    mic = SocketMic()
    data = pcm(range(1024))  # two blocks, sent as odd-sized pieces
    for piece in (data[:1], data[1:700], data[700:1500], data[1500:]):
        mic.push(piece)
    first, second = mic.read(timeout=0), mic.read(timeout=0)
    assert first[1] == pytest.approx(1 / 32768) and second[0] == pytest.approx(512 / 32768)
    assert mic.read(timeout=0) is None


def test_several_blocks_in_one_message():
    mic = SocketMic()
    mic.push(pcm([0] * 512 * 3))
    assert [mic.read(timeout=0) is not None for _ in range(4)] == [True, True, True, False]


def test_drain_discards_queued_blocks():
    mic = SocketMic()
    mic.push(pcm([0] * 1024))
    mic.drain()
    assert mic.read(timeout=0) is None


def test_read_after_close_raises_disconnected():
    mic = SocketMic()
    mic.close()
    with pytest.raises(Disconnected):
        mic.read(timeout=0)


def test_no_audio_for_longer_than_the_idle_timeout_raises_idle_timeout():
    now = [0.0]
    mic = SocketMic(idle_timeout_s=300, clock=lambda: now[0])
    now[0] = 299.0
    assert mic.read(timeout=0) is None
    mic.push(pcm([0] * 512))  # audio restarts the idle clock
    mic.drain()
    now[0] = 598.0
    assert mic.read(timeout=0) is None
    now[0] = 600.0
    with pytest.raises(IdleTimeout):
        mic.read(timeout=0)


def test_start_and_stop_exist_for_the_pipeline():
    mic = SocketMic()
    mic.start()
    mic.stop()


def test_player_sends_16_bit_little_endian_audio():
    sent = []
    SocketPlayer(sent.append).play(np.array([0.0, 0.5, -0.5], dtype=np.float32))
    assert np.frombuffer(sent[0], dtype="<i2").tolist() == [0, 16383, -16383]


def test_player_clips_out_of_range_samples_and_zeroes_nan():
    sent = []
    SocketPlayer(sent.append).play(np.array([2.0, -3.0, np.nan], dtype=np.float32))
    assert np.frombuffer(sent[0], dtype="<i2").tolist() == [32767, -32767, 0]


def test_player_lets_a_send_failure_through():
    def send(data):
        raise Disconnected("closed")

    with pytest.raises(Disconnected):
        SocketPlayer(send).play(np.zeros(4, dtype=np.float32))
