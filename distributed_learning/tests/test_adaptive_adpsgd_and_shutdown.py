import sys
import time
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from volunteer import Volunteer
from manager import Manager
from coordinator import Coordinator
from src.config import PEER_TIMEOUT
from src.adpsgd import BipartiteTopology, ADPSGDStats

class TestAdaptiveADPSGDAndShutdown(unittest.TestCase):
    @patch('volunteer.socket.socket')
    @patch('volunteer.send_message')
    @patch('volunteer.receive_message')
    def test_auto_shutdown_on_no_peers(self, mock_recv, mock_send, mock_sock):
        # Test that a volunteer shuts down when there are no peers
        with patch('volunteer.Volunteer._estimate_resources_vs_needs', return_value={}):
            vol = Volunteer(vol_id=0, my_ip="127.0.0.1")
            vol.peer_timeout = 1 # 1 second for fast test
            vol.last_active_peer_time = time.time() - 2 # 2 seconds ago (already timed out)
            
            # _fetch_active_volunteers returns empty list of candidates
            mock_recv.return_value = ("MSG_NEIGHBORS_RESPONSE", {"volunteers": []}, None)
            
            # Run gossip round
            vol._run_gossip_round()
            
            self.assertFalse(vol._running)

    def test_resource_score_sorting_and_role_assignment(self):
        # Mock Manager for testing dynamic role assignment
        manager = Manager()
        
        # Add a high resource volunteer and a low resource volunteer
        mock_high = MagicMock()
        mock_high.resources.cpu_cores = 16
        mock_high.resources.cpu_freq_ghz = 3.5
        mock_high.resources.ram_gb = 64
        mock_high.resources.network_bandwidth_mbps = 1000
        mock_high.resources.battery = 100
        mock_high.resources.cpu_load = 5
        mock_high.to_dict.return_value = {"mac_address": "mac_high"}
        
        mock_low = MagicMock()
        mock_low.resources.cpu_cores = 2
        mock_low.resources.cpu_freq_ghz = 1.5
        mock_low.resources.ram_gb = 4
        mock_low.resources.network_bandwidth_mbps = 10
        mock_low.resources.battery = 20
        mock_low.resources.cpu_load = 90
        mock_low.to_dict.return_value = {"mac_address": "mac_low"}
        
        manager._volunteers = {
            "mac_high": mock_high,
            "mac_low": mock_low
        }
        manager._neighbor_rewards = {
            "mac_high": [],
            "mac_low": []
        }
        
        # Call _on_neighbors_request for high
        mock_conn = MagicMock()
        with patch('manager.send_message') as mock_send_msg:
            manager._on_neighbors_request(mock_conn, "127.0.0.1", {"volunteer_mac": "mac_high"})
            # Verify high is active
            args, _ = mock_send_msg.call_args
            resp_data = args[2]
            self.assertEqual(resp_data["assigned_role"], "active")
            
            # Call _on_neighbors_request for low
            manager._on_neighbors_request(mock_conn, "127.0.0.2", {"volunteer_mac": "mac_low"})
            args, _ = mock_send_msg.call_args
            resp_data = args[2]
            self.assertEqual(resp_data["assigned_role"], "passive")

if __name__ == '__main__':
    unittest.main()
