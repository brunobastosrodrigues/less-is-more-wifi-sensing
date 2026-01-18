#!/usr/bin/env python3
"""
Link Selection Strategies for Baseline Comparisons

Provides a LinkSelector class to implement various strategies for selecting
or modifying CSI links within a data window. This is crucial for establishing
fair baselines (e.g., single-best-link) to compare against the full multi-link
mesh system.
"""

import logging
import random
from typing import List, Dict, Literal
import numpy as np

logger = logging.getLogger(__name__)

LinkSelectionStrategy = Literal[
    "all",
    "best_variance",
    "average_all",
    "random_single",
    "top_k_variance"
]


class LinkSelector:
    """
    Selects or modifies a list of CSI links based on a specified strategy.
    """

    def __init__(self, strategy: LinkSelectionStrategy = "all", k: int = 5):
        # Basic validation
        valid_strategies = ["all", "best_variance", "average_all", "random_single", "top_k_variance"]
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown link selection strategy: {strategy}")

        self.strategy = strategy
        self.k = k
        logger.debug(f"LinkSelector initialized with strategy: {self.strategy} and k={self.k}")

    def select(self, links: List[Dict]) -> List[Dict]:
        """
        Apply the selection strategy to a list of links from a window.

        Args:
            links: A list of link data dictionaries. Each dict should have
                   'rssi_values', 'csi_frames', etc.

        Returns:
            A new list of links, modified according to the strategy.
        """
        if not links:
            return []

        if self.strategy == "all":
            return links
        elif self.strategy == "best_variance":
            return self._select_best_variance(links)
        elif self.strategy == "average_all":
            return self._select_average_all(links)
        elif self.strategy == "random_single":
            return self._select_random_single(links)
        elif self.strategy == "top_k_variance":
            return self._select_top_k_variance(links)
        else:
            return links  # Should not happen due to __init__ check

    def _select_best_variance(self, links: List[Dict]) -> List[Dict]:
        """Selects the single link with the highest RSSI variance."""
        if len(links) == 1:
            return links

        best_link = max(links, key=lambda link: np.var(link.get('rssi_values', [0])))
        return [best_link]

    def _select_average_all(self, links: List[Dict]) -> List[Dict]:
        """Creates a single 'virtual' link by averaging all links."""
        if len(links) == 1:
            return links

        # Deep copy to avoid modifying original frames in place
        all_csi_frames = [frame for link in links for frame in link.get('csi_frames', [])]
        if not all_csi_frames:
            # Handle case with no CSI data
            avg_rssi = np.mean([link.get('rssi_mean', -100) for link in links])
            total_packets = sum(link.get('packet_count', 0) for link in links)
            return [{
                'tx': 'VIRTUAL', 'rx': 'AVERAGE',
                'rssi_mean': avg_rssi,
                'rssi_values': [],
                'csi_frames': [],
                'packet_count': total_packets
            }]

        # Find the max length of CSI frames
        max_len = max(len(frame) for frame in all_csi_frames)

        # Pad all frames to the max length with their own mean
        padded_frames = []
        for frame in all_csi_frames:
            if len(frame) < max_len:
                padding_value = np.mean(frame) if frame else 0
                padded_frames.append(frame + [padding_value] * (max_len - len(frame)))
            else:
                padded_frames.append(frame)

        # Calculate the mean CSI frame
        avg_csi_frame = np.mean(padded_frames, axis=0).tolist()
        avg_rssi = np.mean([link.get('rssi_mean', -100) for link in links])
        total_packets = sum(link.get('packet_count', 0) for link in links)

        # Collect all RSSI values for variance calculation
        all_rssi_values = [val for link in links for val in link.get('rssi_values', [])]

        virtual_link = {
            'tx': 'VIRTUAL',
            'rx': 'AVERAGE',
            'rssi_mean': avg_rssi,
            'rssi_values': all_rssi_values,
            'csi_frames': [avg_csi_frame] * len(all_csi_frames), # Keep frame count consistent
            'packet_count': total_packets
        }
        return [virtual_link]

    def _select_random_single(self, links: List[Dict]) -> List[Dict]:
        """Selects a single link at random."""
        return [random.choice(links)]

    def _select_top_k_variance(self, links: List[Dict]) -> List[Dict]:
        """Selects the top K links with the highest RSSI variance."""
        if len(links) <= self.k:
            return links

        # Sort links by variance in descending order and take the top k
        sorted_links = sorted(links, key=lambda link: np.var(link.get('rssi_values', [0])), reverse=True)
        return sorted_links[:self.k]
