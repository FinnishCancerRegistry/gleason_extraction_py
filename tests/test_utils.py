import os
import sys
import unittest

import numpy as np
import pandas as pd
import testutils as u

script_file_path = os.path.realpath(__file__)
sys.path.append(
	os.path.dirname(os.path.dirname(os.path.dirname(script_file_path)))
)

import gleason_extraction_py as ge

class TestUtils(unittest.TestCase):

	def test_determine_element_combinations_simple(self):
		dt = pd.DataFrame({	
			'a': [4, np.nan, np.nan],
			'b': [np.nan, 5, np.nan],
			't': [np.nan, np.nan, np.nan],
			'c': [np.nan, np.nan, 9]
			})
		produced = ge.determine_element_combinations(dt)
		expected = pd.DataFrame({	
			'grp': [0.0, 0.0, 0.0],
			'a': [4, np.nan, np.nan],
			'b': [np.nan, 5, np.nan],
			'c': [np.nan, np.nan, 9]
			})
		diff = u.compare_dts(expected, produced, ["grp","a", "b", "c"])
		self.assertTrue(diff.empty)


	def test_determine_element_combinations(self):
		dt = pd.DataFrame({	
			'a': [4, 4, np.nan, np.nan, np.nan, np.nan],
			'b': [np.nan, np.nan, 3, 4, 5, np.nan],
			't': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
			'c': [np.nan, np.nan, np.nan, np.nan,np.nan, 9]
			})
		produced = ge.determine_element_combinations(dt)
		expected = pd.DataFrame({	
			'grp': [0.0, 0.0, 0.0, 0.0 ,1.0 ,2.0],
			'a': [4, 4, np.nan, np.nan, np.nan, np.nan],
			'b': [np.nan, np.nan, 3, 4, 5, np.nan],
			'c': [np.nan, np.nan, np.nan, np.nan,np.nan, 9]
			})
		diff = u.compare_dts(expected, produced, ["grp","a", "b", "c"])
		self.assertTrue(diff.empty)


if __name__ == '__main__':
	unittest.main()