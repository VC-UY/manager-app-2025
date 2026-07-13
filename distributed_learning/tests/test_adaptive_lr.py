import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class TestAdaptiveLR(unittest.TestCase):
    @patch('volunteer.load_dataset')
    @patch('volunteer.create_model')
    @patch('volunteer.ModelProfiler')
    @patch('volunteer.get_resource_info')
    @patch('volunteer.AdvancedProfiler')
    @patch('volunteer.StatsTracker')
    @patch('volunteer.Volunteer._detect_ip')
    @patch('volunteer.Volunteer._estimate_resources_vs_needs')
    def test_adastair_logic(self, mock_est, mock_detect_ip, mock_stats, mock_adv_prof, mock_res_info, mock_model_prof, mock_create_model, mock_load_dataset):
        mock_detect_ip.return_value = "127.0.0.1"
        mock_load_dataset.return_value = (MagicMock(), MagicMock())
        
        from volunteer import Volunteer
        
        # Instantiate a mock volunteer with MAX_ROUNDS = 10
        with patch('volunteer.MAX_ROUNDS', 10), patch('volunteer.LEARNING_RATE', 0.1):
            vol = Volunteer(volunteer_id=0, n_volunteers=2, coordinator_host="127.0.0.1", manager_host="127.0.0.1")
            
            self.assertEqual(vol.current_lr, 0.1)
            self.assertEqual(vol.rstair_rounds, [5, 7, 8])
            
            # Simulate gossip rounds
            for r in range(1, 11):
                vol.round_num = r
                if vol.round_num in vol.rstair_rounds:
                    vol.current_lr = vol.current_lr / 2.0
            
            # Halved 3 times: 0.1 -> 0.05 -> 0.025 -> 0.0125
            self.assertEqual(vol.current_lr, 0.0125)

    @patch('volunteer.load_dataset')
    @patch('volunteer.create_model')
    @patch('volunteer.ModelProfiler')
    @patch('volunteer.get_resource_info')
    @patch('volunteer.AdvancedProfiler')
    @patch('volunteer.StatsTracker')
    @patch('volunteer.Volunteer._detect_ip')
    @patch('volunteer.Volunteer._estimate_resources_vs_needs')
    def test_adaloss_logic(self, mock_est, mock_detect_ip, mock_stats, mock_adv_prof, mock_res_info, mock_model_prof, mock_create_model, mock_load_dataset):
        mock_detect_ip.return_value = "127.0.0.1"
        mock_load_dataset.return_value = (MagicMock(), MagicMock())
        
        from volunteer import Volunteer
        
        with patch('volunteer.MAX_ROUNDS', 10), patch('volunteer.LEARNING_RATE', 0.1):
            vol = Volunteer(volunteer_id=0, n_volunteers=2, coordinator_host="127.0.0.1", manager_host="127.0.0.1")
            
            self.assertEqual(vol.current_lr, 0.1)
            self.assertEqual(vol.rloss_patience, [2, 1, 1])  # 10*0.25=2, 10*0.15=1, 10*0.10=1
            
            # Setup initial state
            vol.adaloss_last_loss = 1.0
            vol.adaloss_counter = 0
            vol.adaloss_alpha = 0
            
            # Round 1: loss decreases to 0.8 -> counter should be 0, last_loss updated to 0.8
            loss_t = 0.8
            patience = vol.rloss_patience[vol.adaloss_alpha] # patience = 2
            if loss_t >= vol.adaloss_last_loss:
                vol.adaloss_counter += 1
            else:
                vol.adaloss_counter = 0
                vol.adaloss_last_loss = loss_t
            
            self.assertEqual(vol.adaloss_counter, 0)
            self.assertEqual(vol.adaloss_last_loss, 0.8)
            self.assertEqual(vol.current_lr, 0.1)
            
            # Round 2: loss increases/stabilizes to 0.9 -> counter should be 1
            loss_t = 0.9
            if loss_t >= vol.adaloss_last_loss:
                vol.adaloss_counter += 1
            else:
                vol.adaloss_counter = 0
                vol.adaloss_last_loss = loss_t
                
            self.assertEqual(vol.adaloss_counter, 1)
            self.assertEqual(vol.current_lr, 0.1)
            
            # Round 3: loss increases/stabilizes to 0.9 -> counter should be 2 (reaches patience of 2)
            loss_t = 0.9
            if loss_t >= vol.adaloss_last_loss:
                vol.adaloss_counter += 1
            else:
                vol.adaloss_counter = 0
                vol.adaloss_last_loss = loss_t
            
            if vol.adaloss_counter >= patience:
                vol.current_lr = vol.current_lr / 2.0
                vol.adaloss_alpha = min(vol.adaloss_alpha + 1, len(vol.rloss_patience) - 1)
                vol.adaloss_counter = 0
                vol.adaloss_last_loss = loss_t
                
            self.assertEqual(vol.adaloss_counter, 0)
            self.assertEqual(vol.adaloss_last_loss, 0.9)
            self.assertEqual(vol.adaloss_alpha, 1) # patience index moves to 1
            self.assertEqual(vol.current_lr, 0.05) # learning rate halved

if __name__ == '__main__':
    unittest.main()
