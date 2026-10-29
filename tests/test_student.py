import unittest
import tempfile
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from levo_harmony.model import MixedSongStudent, MusicConfig
from levo_harmony.contracts import CodecContract, kv_cache_bytes
from levo_harmony.pruning import prune_backbone_depth


class StudentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2307)
        cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                         vocab_size=64, tie_word_embeddings=True)
        self.model = MixedSongStudent(Qwen3ForCausalLM(cfg), MusicConfig(music_codebook_size=16)).eval()

    def test_prefix_and_training_backward(self):
        ids = torch.tensor([[1, 2, 3]])
        styles = torch.tensor([2])
        codes = torch.tensor([[4, 5]])
        logits = self.model.teacher_logits(ids, styles, codes)
        self.assertEqual(logits.shape, (1, 3, 17))
        targets = torch.tensor([[4, 5, 16]])
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.music_head.weight.grad).all())

    def test_untrained_checkpoint_cannot_claim_music(self):
        with self.assertRaisesRegex(RuntimeError, "untrained"):
            self.model.generate_music(torch.tensor([[1]]), torch.tensor([0]))

    def test_length_and_style(self):
        with self.assertRaises(ValueError):
            self.model.prefix(torch.ones(1, 513, dtype=torch.long), torch.tensor([0]))
        with self.assertRaises(ValueError):
            self.model.prefix(torch.tensor([[1]]), torch.tensor([[0]]))

    def test_cached_decode_matches_full_forward(self):
        prefix, _ = self.model.prefix(torch.tensor([[1, 2]]), torch.tensor([1]))
        seq = torch.cat((prefix, self.model.music(torch.tensor([[16, 3, 4]]))), 1)
        full = self.model.backbone.model(inputs_embeds=seq, use_cache=False).last_hidden_state
        first = self.model.backbone.model(inputs_embeds=seq[:, :-1], use_cache=True)
        last = self.model.backbone.model(inputs_embeds=seq[:, -1:], past_key_values=first.past_key_values, use_cache=True).last_hidden_state
        torch.testing.assert_close(full[:, -1], last[:, -1], atol=1e-5, rtol=1e-4)

    def test_contract_and_kv(self):
        with self.assertRaises(ValueError):
            CodecContract("UNVERIFIED").validate()
        c = CodecContract("verified-fixture")
        c.validate()
        self.assertEqual(c.frames(30), 750)
        self.assertEqual(kv_cache_bytes(28, 8, 128, 100), 11468800)

    def test_pretrained_backbone_save_load_is_exact(self):
        # Test the actual from_pretrained path without downloading a full model.
        # Tiny random saved checkpoint, never labeled public pretrained weights.
        with tempfile.TemporaryDirectory() as directory:
            self.model.backbone.save_pretrained(directory)
            loaded = MixedSongStudent.from_pretrained(directory, self.model.config)
            torch.testing.assert_close(loaded.backbone.get_input_embeddings().weight,
                                       self.model.backbone.get_input_embeddings().weight)
            self.assertFalse(loaded.music_trained)

    def test_pruned_cache_indices_remain_valid(self):
        report = prune_backbone_depth(self.model, (0,))
        self.assertEqual(report['remaining_layers'], 1)
        self.assertEqual(self.model.backbone.model.layers[0].self_attn.layer_idx, 0)
        self.test_cached_decode_matches_full_forward()

    def test_pruned_backbone_can_save_and_reload(self):
        prune_backbone_depth(self.model, (0,))
        self.assertEqual(len(self.model.backbone.config.layer_types), 1)
        with tempfile.TemporaryDirectory() as directory:
            self.model.backbone.save_pretrained(directory)
            loaded = MixedSongStudent.from_pretrained(directory, self.model.config)
            self.assertEqual(loaded.backbone.config.num_hidden_layers, 1)
            self.assertEqual(len(loaded.backbone.config.layer_types), 1)
            for name, tensor in self.model.backbone.state_dict().items():
                torch.testing.assert_close(loaded.backbone.state_dict()[name], tensor, atol=0, rtol=0)
            self.assertFalse(loaded.music_trained)

    def test_stream_pause_resume_matches_full_context(self):
        # Only this tiny random fixture bypasses the production training guard.
        self.model.music_trained = True
        head = torch.nn.Linear(self.model.music_head.in_features, self.model.music_head.out_features, bias=True)
        with torch.no_grad():
            head.weight.copy_(self.model.music_head.weight)
            head.bias.zero_()
            head.bias[self.model.eos] = -1000
        self.model.music_head = head
        lyric, style = torch.tensor([[1,2,3]]), torch.tensor([1])
        stream = self.model.stream_music(lyric,style,6)
        prefix, _ = self.model.prefix(lyric,style)
        ids = [self.model.bos]
        paused = False
        for index in range(6):
            with torch.inference_mode():
                inputs = torch.cat((prefix,self.model.music(torch.tensor([ids]))),1)
                hidden = self.model.backbone.model(inputs_embeds=inputs,use_cache=False).last_hidden_state[:, -1]
                expected = self.model.music_head(hidden).argmax(-1).item()
            if expected == self.model.eos:
                with self.assertRaises(StopIteration): next(stream)
                self.assertTrue(stream.ended_by_eos)
                break
            self.assertEqual(next(stream), expected)
            self.assertEqual(stream.cache.get_seq_length(), prefix.shape[1]+len(ids))
            ids.append(expected)
            if index == 2:
                paused = True
                cache = stream.cache
                positions = cache.get_seq_length()
                # Numeric codec work between tokens must not recreate this KV.
                _ = torch.sin(torch.arange(32,dtype=torch.float32))
                self.assertIs(stream.cache,cache)
                self.assertEqual(stream.cache.get_seq_length(),positions)
        list(stream)
        self.assertTrue(paused)
        self.assertTrue(stream.closed)
        self.assertIsNone(stream.cache)

    def test_stream_guard_and_explicit_release(self):
        with self.assertRaisesRegex(RuntimeError,'untrained'):
            self.model.stream_music(torch.tensor([[1]]),torch.tensor([0]))
        self.model.music_trained = True  # Tiny lifecycle fixture only.
        stream = self.model.stream_music(torch.tensor([[1]]),torch.tensor([0]),4)
        stream.close()
        self.assertIsNone(stream.model)
        self.assertIsNone(stream.cache)
        with self.assertRaises(StopIteration): next(stream)

    def test_stream_eos_never_enters_codec_ids(self):
        self.model.music_trained = True  # Tiny protocol fixture only.
        head = torch.nn.Linear(self.model.music_head.in_features,self.model.music_head.out_features,bias=True)
        with torch.no_grad():
            head.weight.zero_();head.bias.zero_();head.bias[self.model.eos] = 1000
        self.model.music_head = head
        stream = self.model.stream_music(torch.tensor([[1]]),torch.tensor([0]),4)
        self.assertEqual(list(stream),[])
        self.assertTrue(stream.ended_by_eos)
        self.assertIsNone(stream.cache)
        self.assertEqual(self.model.generate_music(torch.tensor([[1]]),torch.tensor([0]),4),
                         {'tokens':[],'eos':True})


if __name__ == "__main__":
    unittest.main()
