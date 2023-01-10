import os
import re
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


class TestGleasonExtraction(unittest.TestCase):
	 
	def test_extract_gleason_scores(self):
		text_example = ["gleason 4 + 4 = gleason 8", "gleason 8", "gleason 4 + 4"] 
		text_ids= [0, 1, 2]
		extracted = ge.extract_gleason_scores(text_example, text_ids)
		expected = pd.DataFrame({
								'text_id': [0, 1, 2],	
								'a': [4, np.nan, 4],
								'b': [4, np.nan, 4],
								'c': [8, 8, np.nan]
								})
		diff = u.compare_dts(expected, extracted, ["text_id","a", "b", "c"])
		self.assertTrue(diff.empty, msg = "Did not extract expected values.\n{0}\nLeft_only: expected, but not found.\nRight_only: found, but not expected""".format(diff))


	@unittest.skipIf((not os.path.exists("input.csv") and not os.path.exists("output.csv")), reason="validation data is missing")
	#@unittest.skip
	def test_validation(self):
		input = pd.read_csv("input.csv") #columns: text_id,text
		input = input.groupby("text_id").first().reset_index() # delete duplicates
		texts = list(input.text)
		text_ids = list(input.text_id)
		extracted = ge.extract_gleason_scores(texts, text_ids)
		expected = pd.read_csv("output.csv") #columns: text_id,a,b,c
		diff = u.compare_dts(expected, extracted, ["text_id","a", "b", "c"])
		self.assertTrue(diff.empty, msg = "Did not extract the same values as the validated program.\n{0}\nLeft_only:gleason found by the validated program only.\nRight_only: gleason found by this program only""".format(diff))
		
	def test_parse_gleason_value_string_elements(self):
		value_strings = ["3 + 4 = 7", "7", "3 + 4 (7)", "7 (3 + 4)", "3 + 4", "3 + 4 gleason score 7", "3 + 4 (+5) = 7", "3 + 4 (+5)", "3+4+5", "4+3+5, gleason score 7", "5 4", "3 + 4 / 4 + 3", "3"]
		match_types = ["a + b = c", "c", "a + b = c", "a + b = c", "a + b", "a + b = c", "a + b + t = c", "a + b + t", "a + b + t", "a + b + t = c", "a", "a + b", "kw_all_a"]
		produced = ge.parse_gleason_value_string_elements(value_strings, match_types)
		expected = pd.DataFrame({	
						'pos': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 11, 12],
						'a': [3, np.nan, 3, 3, 3, 3, 3, 3, 3, 4, 5, 4, 3, 4, 3],
						'b': [4, np.nan, 4, 4, 4, 4, 4, 4, 4, 3, np.nan, np.nan, 4, 3, 3],
						't': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5, 5, 5, 5, np.nan, np.nan, np.nan, np.nan, np.nan],
						'c': [7, 7, 7, 7, np.nan, 7, 7, np.nan, np.nan, 7, np.nan, np.nan, np.nan, np.nan, np.nan]
						})
		diff = u.compare_dts(expected, produced, ["pos", "a", "b", "t", "c"])
		self.assertTrue(diff.empty, msg = "\n{0}\nLeft_only:expected\nRight_only: extracted""".format(diff))

	def test_whitelist_to_whitelist_regex(self):
		wh_regex = ge.whitelist_to_whitelist_regex(["hi", "yo"])
		self.assertEqual(re.sub(wh_regex, "", "hi yo hi hi yo"), "")
		self.assertEqual(re.sub(wh_regex, "", "hi yo hiya yoman ho", count = 1), "ya yoman ho")
	
	def test_word_whitelist_to_word_whitelist_regex(self):
		wh_regex = ge.word_whitelist_to_word_whitelist_regex(["hi", "yo"])
		self.assertEqual(re.sub(wh_regex, "", "hi yo hi hi yo"), "")
		self.assertEqual(re.sub(wh_regex, "", "hi yo hiya yoman ho"), "ho")

	def test_multiple_alternative_value_matches(self):
		regex = ge.multiple_alternative_value_matches("[0-9]")
		self.assertEqual(re.search(regex, "primääri gleason gradus (oikea, vasen): 4 5").group(0), "4 5")
		self.assertEqual(re.search(regex, "primääri gleason gradus (oikea, vasen): 4 5 sana").group(0), "4 5")

	def test_some_gleason_extraction_regexes(self):
		self.assertTrue(re.sub(ge.zero_to_three_arbitrary_natural_language_words, "_", "one two three four", count = 1), "_four")
		gleas_regex = ge.whitelist_gleason_word
		self.assertEqual(re.search(gleas_regex, "gleason").group(0), "gleason")
		self.assertEqual(re.search(gleas_regex, "gliisonin").group(0), "gliisonin")
		
		base_regex = ge.whitelist_base_optional_regex
		self.assertEqual(re.search(base_regex, "gleason gradus (3-5) n. 8").group(0), "")
		
		base_gleas_regex = ge.base_gleason_regex
		self.assertEqual(re.search(base_gleas_regex, "gleason gradus (3-5) n. 8").group(0), "gleason gradus (3-5) n. ")
		self.assertEqual(re.sub(base_gleas_regex, "", "gleason lk (1-5) (jotain muuta)"), "")
		self.assertEqual(re.sub(base_gleas_regex, "", "gleason gradus (2-5) (gleasongr2)"), "")
		
		optn_base_gleas_regex = ge.optional_base_gleason_regex
		self.assertEqual(re.search(optn_base_gleas_regex, "gradus gleason (3-5) n. 8").group(0), "gradus gleason (3-5) n. ")
		
		primary_regex = ge.whitelist_primary_regex
		self.assertEqual(re.sub(primary_regex, "", "tavallisin/aggressiivisin"), "")
		self.assertEqual(re.sub(primary_regex, "", "yleisin / aggressiivisin"), "")
		self.assertEqual(re.sub(primary_regex, "", "yleisin"), "")
		self.assertEqual(re.sub(primary_regex, "", "tavallisin"), "")
		self.assertEqual(re.sub(primary_regex, "", "primääri"), "")

		secondary_regex = ge.whitelist_secondary_regex
		self.assertEqual(re.sub(secondary_regex, "", "toiseksi tavallisin/aggressiivisin"), "")
		self.assertEqual(re.sub(secondary_regex, "", "2. tavallisin/aggressiivisin"), "")
		self.assertEqual(re.sub(secondary_regex, "", "2. yleisin / aggressiivisin"), "")
		self.assertEqual(re.sub(secondary_regex, "", "2. yleisin"), "")
		self.assertEqual(re.sub(secondary_regex, "", "toiseksi tavallisin"), "")
		self.assertEqual(re.sub(secondary_regex, "", "sekundääri"), "")
		
		scoresum_regex = ge.whitelist_scoresumword_regex
		self.assertEqual(re.search(scoresum_regex, "yht.pist.").group(0), "yht.pist.")
		self.assertEqual(re.search(scoresum_regex, "pistesumma").group(0), "pistesumma")

	def test_some_addition_dt_regexes(self):	
		self.assertEqual(re.search(ge.a_plus_b, "3 + 3").group(0), "3 + 3")
		self.assertEqual(re.search(ge.a_plus_b, "3+5").group(0), "3+5")
		self.assertFalse(bool(re.search(ge.a_plus_b, "3 ja 3")))
		self.assertEqual(re.search(ge.a_comma_b, "gleason 7 (3,4)").group(0), "3,4")
		self.assertEqual(re.search(ge.a_plus_b_plus_t, "3 + 3 + 3").group(0), "3 + 3 + 3")
		self.assertEqual(re.search(ge.a_plus_b_plus_t, "3+5(+4)").group(0), "3+5(+4)")
		self.assertFalse(bool(re.search(ge.a_plus_b_plus_t, "3 + 5 4")))
		self.assertEqual(re.search(ge.a_comma_b_comma_t, "gleason 7 (3,4,4)").group(0), "3,4,4")


		self.assertEqual(re.sub(ge.addition_values[1], "", "7 = 3 + 4"), "")

	def test_some_keyword_dt_regexes(self):
		self.assertTrue(u.is_substituted(ge.addition_guide, "(a + b)"))
		self.assertTrue(u.is_substituted(ge.addition_guide, "x + y"))

		self.assertTrue(u.is_found(ge.kw_c_prefix, "gleason yht.pist.", "gleason yht.pist."))
		self.assertTrue(u.is_found(ge.kw_c_prefix, "gleason pistesumma", "gleason pistesumma"))
		self.assertFalse(u.is_found(ge.kw_c_prefix, "pelkästään gleason 1"))
		self.assertTrue(u.is_found(ge.kw_c_prefix, "gleason score, summa a+b (2-10) 7", "gleason score, summa a+b (2-10) "))

		self.assertFalse(u.is_found(ge.whitelist_tertiary_regex, "3.tblyleisin gleason-gradus (1-5) 5"))

	def test_prepare_text(self):
		self.assertEqual(ge.prepare_text("Gleason 7 (4+3)"), "gleason 7 (4+3)")
		self.assertEqual(ge.prepare_text("Is bad (Gleason score 9-10): no"), "is bad no")

	
if __name__ == '__main__':
	unittest.main()
