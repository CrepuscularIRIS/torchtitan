import unittest
from unittest.mock import MagicMock
import torch

from torchtitan.trainer import Trainer

class TestTokensSeen(unittest.TestCase):
    def test_cp_token_normalization(self):
        """
        Verify that ntokens_batch is properly divided by cp_degree 
        inside trainer.batch_generator to prevent metric inflation.
        """
        seq_len = 100
        global_batch_size = 8
        labels = torch.zeros(global_batch_size, seq_len)
        actual_tokens_in_batch = labels.numel() # 800
        
        for cp_degree in [1, 2, 4]:
            # 1. Mock the trainer and its dependencies
            mock_trainer = MagicMock(spec=Trainer)
            mock_trainer.ntokens_seen = 0
            
            # 2. Mock ParallelDims with our cp_degree
            mock_trainer.parallel_dims = MagicMock()
            mock_trainer.parallel_dims.cp = cp_degree
            
            # 3. Mock the metrics processor 
            mock_metrics = MagicMock()
            mock_metrics.ntokens_since_last_log = 0
            mock_metrics.data_loading_times = []
            mock_trainer.metrics_processor = mock_metrics
            
            # Dummy dataloader yielding our batch once
            data_iterable = [({"input": torch.zeros(1)}, labels)]
            
            # 4. Extract and run the unbound method, passing the mock as 'self'
            generator = Trainer.batch_generator(mock_trainer, data_iterable)
            next(generator)
            
            # 5. Verify the token counts incremented correctly 
            expected_normalized_tokens = actual_tokens_in_batch // cp_degree
            self.assertEqual(mock_trainer.ntokens_seen, expected_normalized_tokens)
            
            # 6. Verify mathematically that dist_sum over CP ranks reconstructs actual
            simulated_dist_sum = mock_trainer.ntokens_seen * cp_degree
            self.assertEqual(simulated_dist_sum, actual_tokens_in_batch)

if __name__ == "__main__":
    unittest.main()
