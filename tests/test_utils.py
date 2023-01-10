import unittest
import numpy as np
import pandas as pd
import utils
import testutils as u


class TestUtils(unittest.TestCase):

	def test_typed_format_dt_to_standard_format_dt(self):
		dt = pd.DataFrame({	
				'text_id': [0, 0, 0],
				'obs_id': [1, 2, 3],
				'match_type': ["a", "b", "a"],
				'a': [3, np.nan, 2],
				'b': [np.nan, 4, np.nan],
				't': [np.nan, np.nan, np.nan],
				'c': [np.nan, np.nan, np.nan],
				'warning': [np.nan, np.nan, np.nan]
				})
		produced = utils.typed_format_dt_to_standard_format_dt(dt)
		expected = pd.DataFrame({	
				'text_id': [0, 0],
				'obs_id': [1, 3],
				'a': [3, 2],
				'b': [4, np.nan],
				'c': [np.nan, np.nan]
				})
		diff = u.compare_dts(expected, produced, ["text_id","obs_id","a", "b", "c"])
		self.assertTrue(diff.empty)
		

	def test_determine_element_combinations_simple(self):
		dt = pd.DataFrame({	
			'a': [4, np.nan, np.nan],
			'b': [np.nan, 5, np.nan],
			't': [np.nan, np.nan, np.nan],
			'c': [np.nan, np.nan, 9]
			})
		produced = utils.determine_element_combinations(dt)
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
		produced = utils.determine_element_combinations(dt)
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