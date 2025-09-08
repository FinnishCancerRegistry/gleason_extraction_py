import os
import regex as re
import sys
import unittest
import numpy as np
import pandas as pd

script_file_path = os.path.realpath(__file__)
sys.path.append(
	os.path.dirname(script_file_path)
)
import unit_test_utils as u
sys.path.append(
	os.path.dirname(os.path.dirname(script_file_path))
)
from src import gleason_extraction as ge
from src import gleason_extraction_regexes as ger

class TestGleasonExtraction(unittest.TestCase):
	 
	def test_extract_gleason_scores(self):
		data = pd.DataFrame({
			"id": list(range(5)),
			"text": [
				"gleason 4 + 4 = gleason 8",
				"gleason 8",
				"gleason 4 + 4",
				"gleason primääri 4\ngleason sekundääri 3",
				"gleason primääri 4\ngleason sekundääri 3\ngleason primääri 3\ngleason sekundääri 4"
			]
		})
		obs = ge.extract_gleason_scores(data["text"], data["id"])
		exp = pd.DataFrame({
			'text_id': [0, 1, 2, 3, 4, 4],	
			'a': [4, np.nan, 4, 4, 4, 3],
			'b': [4, np.nan, 4, 3, 3, 4],
			'c': [8, 8, np.nan, np.nan, np.nan, np.nan]
		}, dtype = "Int64")
		diff = u.compare_dts(exp, obs, ["text_id","a", "b", "c"])
		if not diff.empty:
			print(diff)
			import pdb; pdb.set_trace()
		self.assertTrue(diff.empty)


	@unittest.skipIf((not os.path.exists("tests/data/input.csv") and not os.path.exists("tests/data/output.csv")), reason="validation data is missing")
	#@unittest.skip
	def test_validation(self):
		input = pd.read_csv("tests/data/input.csv") #columns: text_id,text
		input = input.groupby("text_id").first().reset_index() # delete duplicates
		texts = list(input.text)
		text_ids = list(input.text_id)
		extracted = ge.extract_gleason_scores(texts, text_ids)
		expected = pd.read_csv("tests/data/output.csv") #columns: text_id,a,b,c
		diff = u.compare_dts(expected, extracted, ["text_id","a", "b", "c"])
		if not diff.empty:
			print(diff)
			import pdb; pdb.set_trace()
		self.assertTrue(diff.empty)

	def test_whitelist_to_whitelist_regex(self):
		wh_regex = ger.whitelist_to_whitelist_regex(["hi", "yo"])
		self.assertEqual(re.sub(wh_regex, "", "hi yo hi hi yo"), "")
		self.assertEqual(re.sub(wh_regex, "", "hi yo hiya yoman ho", count = 1), "ya yoman ho")
	
	def test_word_whitelist_to_word_whitelist_regex(self):
		wh_regex = ger.word_whitelist_to_word_whitelist_regex(["hi", "yo"])
		self.assertEqual(re.sub(wh_regex, "", "hi yo hi hi yo"), "")
		self.assertEqual(re.sub(wh_regex, "", "hi yo hiya yoman ho"), "ho")

	def test_multiple_alternative_value_matches(self):
		regex = ger.multiple_alternative_value_matches("[0-9]")
		self.assertEqual(re.search(regex, "primääri gleason gradus (oikea, vasen): 4 5").group(0), "4 5") # type: ignore
		self.assertEqual(re.search(regex, "primääri gleason gradus (oikea, vasen): 4 5 sana").group(0), "4 5") # type: ignore

	def test_some_gleason_extraction_regexes(self):
		self.assertTrue(re.sub(ger.zero_to_three_arbitrary_natural_language_words, "_", "one two three four", count = 1), "_four")
		gleas_regex = ger.whitelist_gleason_word
		self.assertEqual(re.search(gleas_regex, "gleason").group(0), "gleason") # type: ignore
		self.assertEqual(re.search(gleas_regex, "gliisonin").group(0), "gliisonin") # type: ignore
		
		base_regex = ger.whitelist_base_optional_regex
		self.assertEqual(re.search(base_regex, "gleason gradus (3-5) n. 8").group(0), "") # type: ignore
		
		base_gleas_regex = ger.base_gleason_regex
		self.assertEqual(re.search(base_gleas_regex, "gleason gradus (3-5) n. 8").group(0), "gleason gradus (3-5) n. ") # type: ignore
		self.assertEqual(re.sub(base_gleas_regex, "", "gleason lk (1-5) (jotain muuta)"), "")
		self.assertEqual(re.sub(base_gleas_regex, "", "gleason gradus (2-5) (gleasongr2)"), "")
		
		optn_base_gleas_regex = ger.optional_base_gleason_regex
		self.assertEqual(re.search(optn_base_gleas_regex, "gradus gleason (3-5) n. 8").group(0), "gradus gleason (3-5) n. ") # type: ignore
		
		primary_regex = ger.whitelist_primary_regex
		self.assertEqual(re.sub(primary_regex, "", "tavallisin/aggressiivisin"), "")
		self.assertEqual(re.sub(primary_regex, "", "yleisin / aggressiivisin"), "")
		self.assertEqual(re.sub(primary_regex, "", "yleisin"), "")
		self.assertEqual(re.sub(primary_regex, "", "tavallisin"), "")
		self.assertEqual(re.sub(primary_regex, "", "primääri"), "")

		secondary_regex = ger.whitelist_secondary_regex
		self.assertEqual(re.sub(secondary_regex, "", "toiseksi tavallisin/aggressiivisin"), "")
		self.assertEqual(re.sub(secondary_regex, "", "2. tavallisin/aggressiivisin"), "")
		self.assertEqual(re.sub(secondary_regex, "", "2. yleisin / aggressiivisin"), "")
		self.assertEqual(re.sub(secondary_regex, "", "2. yleisin"), "")
		self.assertEqual(re.sub(secondary_regex, "", "toiseksi tavallisin"), "")
		self.assertEqual(re.sub(secondary_regex, "", "sekundääri"), "")
		
		scoresum_regex = ger.whitelist_scoresumword_regex
		self.assertEqual(re.search(scoresum_regex, "yht.pist.").group(0), "yht.pist.") # type: ignore
		self.assertEqual(re.search(scoresum_regex, "pistesumma").group(0), "pistesumma") # type: ignore

	def test_some_addition_dt_regexes(self):	
		self.assertEqual(re.search(ger.a_plus_b, "3 + 3").group(0), "3 + 3") # type: ignore
		self.assertEqual(re.search(ger.a_plus_b, "3+5").group(0), "3+5") # type: ignore
		self.assertFalse(bool(re.search(ger.a_plus_b, "3 ja 3")))
		self.assertEqual(re.search(ger.a_comma_b, "gleason 7 (3,4)").group(0), "3,4") # type: ignore
		self.assertEqual(re.search(ger.a_plus_b_plus_t, "3 + 3 + 3").group(0), "3 + 3 + 3") # type: ignore
		self.assertEqual(re.search(ger.a_plus_b_plus_t, "3+5(+4)").group(0), "3+5(+4)") # type: ignore
		self.assertFalse(bool(re.search(ger.a_plus_b_plus_t, "3 + 5 4")))
		self.assertEqual(re.search(ger.a_comma_b_comma_t, "gleason 7 (3,4,4)").group(0), "3,4,4") # type: ignore


		self.assertEqual(re.sub(ger.addition_values[1], "", "7 = 3 + 4"), "")

	def test_some_keyword_dt_regexes(self):
		self.assertTrue(u.is_substituted(ger.addition_guide, "(a + b)"))
		self.assertTrue(u.is_substituted(ger.addition_guide, "x + y"))

		self.assertTrue(u.is_found(ger.kw_c_prefix, "gleason yht.pist.", "gleason yht.pist."))
		self.assertTrue(u.is_found(ger.kw_c_prefix, "gleason pistesumma", "gleason pistesumma"))
		self.assertFalse(u.is_found(ger.kw_c_prefix, "pelkästään gleason 1"))
		self.assertTrue(u.is_found(ger.kw_c_prefix, "gleason score, summa a+b (2-10) 7", "gleason score, summa a+b (2-10) "))

		self.assertFalse(u.is_found(ger.whitelist_tertiary_regex, "3.tblyleisin gleason-gradus (1-5) 5"))

	def test_prepare_text(self):
		self.assertEqual(ge.prepare_text("Gleason 7 (4+3)"), "gleason 7 (4+3)")
		self.assertEqual(ge.prepare_text("Is bad (Gleason score 9-10): no"), "is bad no")

	
if __name__ == '__main__':
	unittest.main()
