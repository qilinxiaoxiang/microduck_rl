import threading
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from voice_fixture_compare import compare_fixture
from voice_capture_adapter import VoiceCaptureAdapter
from voice_shadow import ASR_HZ, INPUT_HZ, StreamingDecimator, VoiceShadow, local_intent, to_16k
from voice_shadow_replay import replay


class VoiceShadowTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.text = "Mochi, fetch"
        self.samples = None

        def transcribe(samples):
            self.samples = samples
            self.started.set()
            if not self.release.wait(3):
                raise TimeoutError("test ASR was not released")
            return self.text

        self.voice = VoiceShadow(transcribe, self.events.append)
        self.t = 0.0
        self.block = np.ones((960, 4), dtype=np.float32) * 0.2

    def tearDown(self):
        self.release.set()
        self.voice.close()

    def feed(self, count=1, **kwargs):
        for i in range(count):
            self.voice.feed(self.t, self.block, **(kwargs if i == 0 else {}))
            self.t += len(self.block) / INPUT_HZ

    def submit(self):
        self.feed(wake_time=self.t + 0.02)
        self.feed(75)
        self.assertTrue(self.started.wait(2))

    def finish(self, now=None):
        self.release.set()
        self.voice.pending[0].result(timeout=3)
        self.voice.poll(self.t if now is None else now)

    def test_same_utterance_and_preroll(self):
        self.feed(20)
        self.submit()
        self.finish()
        result = [e for e in self.events if e["state"] == "intent"]
        self.assertEqual(result[0]["intent"], "fetch")
        self.assertEqual(len(self.samples), int(1.92 * ASR_HZ))
        self.assertNotIn("transcript", result[0])

    def test_preroll_and_deadline_at_block_start(self):
        self.feed(100)
        self.feed(wake_time=self.t)
        self.feed(75)
        self.assertTrue(self.started.wait(2))
        self.finish()
        self.assertEqual(len(self.samples), int(2.5 * ASR_HZ))

    def test_wake_required_and_stopped_supported(self):
        self.text = "stopped"
        self.voice.set_fetch_active(True)
        self.feed(90)
        self.assertFalse(any(e["state"] == "intent" for e in self.events))
        self.submit()
        self.finish()
        self.assertEqual([e for e in self.events if e["state"] == "intent"][-1]["intent"], "stop")
        for text in ("Mochi", "do not stop", "the bus stopped", "fetch or stop"):
            self.assertEqual(local_intent(text), "none")

    def test_active_fetch_cannot_be_replaced(self):
        self.voice.set_fetch_active(True)
        self.submit()
        self.finish()
        self.assertEqual(self.events[-1]["state"], "ignored")
        self.assertIn("already active", self.events[-1]["reason"])

    def test_manual_takeover_invalidates_inflight_result(self):
        self.submit()
        self.voice.invalidate("manual takeover")
        self.finish()
        self.assertEqual(self.events[-1]["state"], "expired")

    def test_late_result_is_not_intent(self):
        self.submit()
        self.finish(now=self.t + 4)
        self.assertEqual(self.events[-1]["state"], "expired")

    def test_worker_overload_reported_and_old_session_invalidated(self):
        self.submit()
        self.feed(wake_time=self.t + 0.02)
        self.feed(75)
        self.assertTrue(any("worker busy" in e.get("error", "") for e in self.events))
        self.finish()
        self.assertEqual(self.events[-1]["state"], "expired")

    def test_playback_suppresses_wake_and_invalidates_pending(self):
        self.submit()
        self.feed(playback=True)
        self.finish()
        self.assertEqual(self.events[-1]["state"], "expired")
        self.feed(playback=True, wake_time=self.t + 0.02)
        self.assertEqual(sum(e["state"] == "listening" for e in self.events), 1)

    def test_playback_invalidates_a_ready_result_before_polling(self):
        self.submit()
        self.release.set()
        self.voice.pending[0].result(timeout=3)
        self.feed(playback=True)
        self.assertEqual(self.events[-1]["state"], "expired")
        self.assertFalse(any(e["state"] == "intent" for e in self.events))

    def test_capture_gap_invalidates_session_and_raises(self):
        self.feed(wake_time=self.t + 0.02)
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            self.voice.feed(self.t + 1, self.block)
        self.assertEqual(self.events[-1]["reason"], "capture discontinuity")

    def test_explicit_capture_restart_invalidates_pending(self):
        self.submit()
        self.voice.restart_capture("device restarted")
        self.finish()
        self.assertEqual(self.events[-1]["state"], "expired")
        self.voice.feed(123.0, self.block)

    def test_invalid_or_empty_asr_is_explicit_error(self):
        self.text = ""
        self.submit()
        self.finish()
        self.assertEqual(self.events[-1]["state"], "error")
        self.assertIn("ASR failed", self.events[-1]["error"])

    def test_decimator_preserves_tone_and_blocks_alias(self):
        t = np.arange(INPUT_HZ) / INPUT_HZ
        def tone(freq):
            x = np.sin(2 * np.pi * freq * t).astype(np.float32)
            return np.column_stack((x, x))
        low = to_16k(tone(1000))[100:]
        high = to_16k(tone(12000))[100:]
        self.assertEqual(len(to_16k(tone(1000))), ASR_HZ)
        self.assertGreater(float(np.sqrt(np.mean(low ** 2))), 0.65)
        self.assertLess(float(np.sqrt(np.mean(high ** 2))), 0.01)

    def test_streaming_resampling_matches_one_shot_across_irregular_blocks(self):
        samples = np.random.default_rng(42).normal(size=(10303, 2)).astype(np.float32)
        decimator = StreamingDecimator()
        streamed = []
        for start in range(0, len(samples), 257):
            streamed.append(decimator.process(samples[start:start + 257])[1])
        np.testing.assert_array_equal(np.concatenate(streamed), to_16k(samples))

    def test_replay_uses_only_labelled_wake_and_rejects_short_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(INPUT_HZ)
                wav.writeframes(np.zeros((INPUT_HZ * 2, 2), dtype="<i2").tobytes())
            events = replay(path, 0.25, lambda _: "stopped")
            self.assertEqual([e["intent"] for e in events if e["state"] == "intent"], ["stop"])
            with self.assertRaisesRegex(ValueError, "before the wake window"):
                replay(path, 1.0, lambda _: "stop")
            with self.assertRaisesRegex(ValueError, "outside the WAV"):
                replay(path, 3.0, lambda _: "stop")


class FixtureCompareTests(unittest.TestCase):
   def test_cloud_is_comparison_only_and_local_decision_wins(self):
       result = compare_fixture("stop", lambda _: "fetch")
       self.assertEqual(result["intent"], "stop")
       self.assertEqual(result["cloud"], {"status": "ok", "intent": "fetch"})
       self.assertEqual(compare_fixture("stopped")["cloud"]["status"], "disabled")

   def test_cloud_failure_is_explicit_and_unknown_fixture_is_rejected(self):
       def fail(_):
           raise TimeoutError("do not expose request data")

       result = compare_fixture("fetch", fail)
       self.assertEqual(result["intent"], "fetch")
       self.assertEqual(result["cloud"], {"status": "error", "error": "TimeoutError"})
       with self.assertRaisesRegex(ValueError, "unknown"):
           compare_fixture("private transcript", fail)


class CaptureAdapterTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.started = threading.Event()
        self.release = threading.Event()

        def transcribe(_):
            self.started.set()
            if not self.release.wait(3):
                raise TimeoutError("test ASR was not released")
            return "Mochi, fetch"

        self.shadow = VoiceShadow(transcribe, self.events.append)
        self.adapter = VoiceCaptureAdapter(self.shadow)
        self.block = np.zeros((960, 4), dtype=np.float32)
        self.t = 100.0004

    def tearDown(self):
        self.release.set()
        self.adapter.close()

    def feed(self, hit=None, playback=False, lag=0):
        end = self.t + len(self.block) / INPUT_HZ
        self.adapter.on_block(self.t, self.block, accepted_hit=hit, playback=playback, now=end + lag)
        self.t = end

    def submit(self):
        # A historical OwnSpotter hit rounds the block end to milliseconds.
        hit = {"keyword": "MOCHI", "t_hit": round(self.t + 0.02, 3), "repeat": False}
        self.feed(hit)
        for _ in range(75):
            self.feed()
        self.assertTrue(self.started.wait(2))
        return hit

    def test_shared_stream_accepts_rounded_wake_and_preserves_input(self):
        original = self.block.copy()
        hit = self.submit()
        self.release.set()
        self.shadow.pending[0].result(timeout=3)
        self.feed()
        self.assertEqual(hit["keyword"], "MOCHI")
        np.testing.assert_array_equal(self.block, original)
        self.assertEqual([e["intent"] for e in self.events if e["state"] == "intent"], ["fetch"])

    def test_old_hit_and_delayed_block_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "not from this capture block"):
            self.feed({"keyword": "MOCHI", "t_hit": self.t - 0.8})
        self.assertEqual(self.events[-1]["reason"], "invalid or delayed capture input")
        with self.assertRaisesRegex(ValueError, "too delayed"):
            self.feed(lag=0.3)

    def test_playback_does_not_open_a_wake_window(self):
        hit = {"keyword": "MOCHI", "t_hit": round(self.t + 0.02, 3)}
        self.feed(hit, playback=True)
        self.assertFalse(any(e["state"] == "listening" for e in self.events))

    def test_capture_loss_and_manual_takeover_invalidate_inflight_work(self):
        self.submit()
        self.adapter.manual_takeover()
        self.release.set()
        self.shadow.pending[0].result(timeout=3)
        self.feed()
        self.assertEqual(self.events[-1]["state"], "expired")
        self.adapter.capture_lost()
        self.assertEqual(self.events[-1]["reason"], "capture lost")

    def test_arrival_clock_expires_old_result(self):
        self.submit()
        for _ in range(145):
            self.feed()
        self.release.set()
        self.shadow.pending[0].result(timeout=3)
        self.feed(lag=0.2)
        self.assertEqual(self.events[-1]["state"], "expired")


if __name__ == "__main__":
    unittest.main()
