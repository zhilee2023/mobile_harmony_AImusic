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


if __name__ == "__main__":
    unittest.main()
