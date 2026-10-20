"""Shape, stacking, and gradient checks for the 3-layer ConvLSTM."""

import unittest

import torch

from convlstm import CLSTM_cell, ConvLSTM


class ConvLSTMTests(unittest.TestCase):
    def test_default_is_three_layers(self):
        model = ConvLSTM(input_dim=1, hidden_dim=8, kernel_size=3)
        self.assertEqual(model.num_layers, 3)
        self.assertEqual(len(model.cell_list), 3)
        self.assertEqual(model.hidden_dim, [8, 8, 8])

    def test_layer_channel_wiring(self):
        model = ConvLSTM(input_dim=2, hidden_dim=[4, 8, 16], kernel_size=3)
        self.assertEqual(model.cell_list[0].input_dim, 2)
        self.assertEqual(model.cell_list[0].hidden_dim, 4)
        self.assertEqual(model.cell_list[1].input_dim, 4)
        self.assertEqual(model.cell_list[1].hidden_dim, 8)
        self.assertEqual(model.cell_list[2].input_dim, 8)
        self.assertEqual(model.cell_list[2].hidden_dim, 16)

    def test_forward_shapes(self):
        seq_len, batch, channels, height, width = 5, 2, 1, 16, 16
        hidden_dim = [8, 12, 6]
        model = ConvLSTM(input_dim=channels, hidden_dim=hidden_dim, kernel_size=3)
        inputs = torch.randn(seq_len, batch, channels, height, width)

        output, states = model(inputs)

        self.assertEqual(tuple(output.shape), (seq_len, batch, hidden_dim[-1], height, width))
        self.assertEqual(len(states), 3)
        for layer_idx, (h, c) in enumerate(states):
            expected = (batch, hidden_dim[layer_idx], height, width)
            self.assertEqual(tuple(h.shape), expected)
            self.assertEqual(tuple(c.shape), expected)

    def test_cell_matches_notebook_calling_convention(self):
        cell = CLSTM_cell(input_dim=3, hidden_dim=5, kernel_size=3)
        x = torch.randn(2, 3, 7, 9)
        hidden = cell.init_hidden(batch_size=2, image_size=(7, 9))
        hy, hidden_state = cell(x, hidden)
        self.assertEqual(tuple(hy.shape), (2, 5, 7, 9))
        self.assertEqual(len(hidden_state), 2)
        self.assertTrue(torch.equal(hy, hidden_state[0]))

    def test_gradients_flow_through_all_layers(self):
        model = ConvLSTM(input_dim=1, hidden_dim=[4, 6, 8], kernel_size=3)
        inputs = torch.randn(4, 2, 1, 8, 8, requires_grad=True)
        output, _ = model(inputs)
        output.mean().backward()

        self.assertIsNotNone(inputs.grad)
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)
        for layer_idx, cell in enumerate(model.cell_list):
            self.assertIsNotNone(cell.conv.weight.grad, msg=f"layer {layer_idx} has no grad")
            self.assertGreater(cell.conv.weight.grad.abs().sum().item(), 0.0)

    def test_optional_external_hidden_state(self):
        model = ConvLSTM(input_dim=1, hidden_dim=4, kernel_size=3, num_layers=3)
        inputs = torch.randn(3, 1, 1, 5, 5)
        hidden_state = model._init_hidden(batch_size=1, image_size=(5, 5))
        output, states = model(inputs, hidden_state=hidden_state)
        self.assertEqual(tuple(output.shape), (3, 1, 4, 5, 5))
        self.assertEqual(len(states), 3)

    def test_hidden_dim_length_must_match_layers(self):
        with self.assertRaises(ValueError):
            ConvLSTM(input_dim=1, hidden_dim=[8, 8], kernel_size=3, num_layers=3)


if __name__ == "__main__":
    unittest.main()
