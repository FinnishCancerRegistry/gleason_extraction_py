import os
import sys
import unittest

import numpy as np
import pandas as pd

script_file_path = os.path.realpath(__file__)
sys.path.append(
	os.path.dirname(os.path.dirname(os.path.dirname(script_file_path)))
)

import gleason_extraction_py as ge


class TestPatternExtraction(unittest.TestCase):

	def test_extract_context_affixed_values(self):
		pattern_dt_example = pd.DataFrame({
			'pattern_name': ['a','b','c'],
			'prefix': ['primary grade[ ]*','secondary grade[ ]*','gleason score[ ]*'],
			'value': ['[3-5]','[3-5]','[6-90]'],
			'suffix': ['', '', '']
		})

		example_texts = ['primary grade 3','secondary grade 4','primary grade 5 secondary grade 3','secondary grade 3 and primary grade 5, therefore gleason score 8','primary grade 5 gleason score 8'] 

		result_df = ge.extract_context_affixed_values(example_texts, pattern_dt_example)
		self.assertEqual(result_df.shape[0], 9)
		self.assertTrue(np.array_equal(result_df['pos'].values, np.array([0,1,2,2,3,3,3,4,4])))
		self.assertTrue(np.array_equal(result_df['pattern_name'].values, np.array(['a', 'b', 'a', 'b', 'b','a','c','a','c'])))
		self.assertTrue(np.array_equal(result_df['value'].values, np.array(['3','4','5','3','3','5','8','5','8'])))


	
	def test_extract_context_affixed_values_only_one_match_due_to_overwriting(self):
		test_pattern_dt = pd.DataFrame({
			'pattern_name': ['secondary','primary'],
			'prefix': ['second most prevalent grade ','most prevalent grade '],
			'value': ['[3-5]','[3-5]'],
			'suffix': ['', '']
		})

		test_text = ['second most prevalent grade 3']

		result_df = ge.extract_context_affixed_values(test_text, test_pattern_dt)
		self.assertTrue(np.array_equal(result_df['value'].values, np.array(['3'])))



if __name__ == '__main__':
	unittest.main()