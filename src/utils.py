import os
import regex as re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import gleason_extraction_regexes as ger
import logs
logger = logs.logging.getLogger('utils')


def normalise_text(x : str) -> str:
	x = re.sub("\\n|\\r", " ", x)
	x = re.sub("[: ]{1,}", " ", x)
	x = re.sub("\\.{2,}", " ", x)
	x = re.sub("\\_+", " ", x)
	x = re.sub("\\-{2,}", " ", x)
	x = re.sub("(?<=[0-9])(?=[a-zåäöA-ZÅÄÖ])", " ", x)
	roman_numerals = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
	roman_numeral_res = [" " + value.upper() + " " for value in roman_numerals]
	
	for idx, val in enumerate(roman_numeral_res):
		x = re.sub(val, " " + str(idx + 1) + " ", x)
	x = re.sub("\\s+", " ", x)
	x = x.lower()
	return x


def determine_element_combinations(
		dt : pd.DataFrame,
		n_max_each : int = 5
	) -> pd.Series:
	"""Combinations of Gleason Score Elements

	Identify Gleason score elements that belong together.
	This Function keeps the dt as it is and adds a column `grp`.

	Args:
		dt (DataFrame): with (at least) columns
			`a`(int/float): primary Gleason score
			`b`(int/float): second Gleason score
			`t`(int/float): tertiary gleason
			`c`(int/float): Gleason scoresum
		`n_max_each` (int, optional): maximum number of times each element type can be repeat. Defaults to 5.

	The problem this functions is intended to solve is the situation where
	multiple Gleason score elements (e.g. `{A, A, B, B, C}`) have been collected
	from a single text, and we wish to assign individual elements into groups
	of elements, where a group of elements can be used to construct  standard 
	form presentations of Gleason scores. E.g. in `{A, A, B, B, C}` we wish
	to identify the 1st, 3rd, and 5th elements as belonging in the same group,
	and the 2nd and 4th as belonging in their own (`{{A,B,C}, {A,B}}`).

	This function contains a fixed list of allowed combinations which are 
	interpreted as belonging in the same group:
	`{C, A, B, T}`
	`{C, A, B}`
	`{C, B, A}`
	`{A, B, T, C}`
	`{A, B, C}`
	`{B, A, C}`
	`{A, B, T}`
	`{A, B}`
	`{A}`
	`{B}`
	`{T}`
	`{C}`
	this list is looped over in the given order. At each combination, it is 
	tested whether `rep(combination, each = n_each)` matches the the observed
	types (inferred internally; here `combination` can be e.g. `{C, A, B}`).
	`n_max_each` determines how many each-style repeats are allowed at most,
	i.e. `n_each` is the changing variable in a for loop over `1:n_max_each`.
	The for loop over `1:n_max_each` is run until a match is found, if any.
	If no match is found for that `combination`, the next one is tried.
	Note that the last three allowed combinations mean that if an element cannot
	be grouped with anything else, it will be the sole member of its own group.

	Raises:
		ValueError: Input dt should NOT have columns grp, grp_type or type, as these are added by this function into dt
		ValueError: There has to be only one value a/b/t/c per row
		
	Returns:
		pd.Series[pd.Int64DType]: Group indicators, one for each row of `dt`.
	"""

	try:
		if not dt.loc[dt[['a','b','t','c']].notnull().sum(axis=1) != 1].empty:
			raise ValueError('There has to be exactly one non-nan a/b/t/c value per row')
	except Exception as e:
		logger.exception(e)
		raise
	

	allowed_combinations = [
		["c", "a", "b", "t"],
		["c", "a", "b"],
		["c", "b", "a"],
		["a", "b", "t", "c"],
		["a", "b", "c"],
		["b", "a", "c"],
		["a", "b", "t"],
		["a", "b"],
		"a",
		"b",
		"t",
		"c"
	]
	dt.reset_index(drop=True, inplace=True)
	groups = pd.Series([None] * dt.shape[0], dtype = "Int64")
	types = []
	for i in range(dt.shape[0]):
		for value_col_nm in ["a", "b", "t", "c"]:
			tmp = dt[value_col_nm].isnull()
			try:
				if not tmp.iloc[i]:
					types.append(value_col_nm)
					break
			except:
				import IPython; IPython.embed()
	types = pd.Series(types)
	n = dt.shape[0]
	max_grp = 0
	while groups.isnull().any():
		wh_first = int(groups.isnull().idxmax())
		for i in range(len(allowed_combinations)):
			break_search = False
			for n_each in range(1, n_max_each + 1):
				candidate = np.repeat(allowed_combinations[i], n_each)
				compare_idx = np.arange(wh_first, min(n, wh_first + len(candidate)))
				if np.array_equal(np.array(types.iloc[compare_idx]), candidate):
					grp_indicators = np.tile(
						np.arange(max_grp, max_grp + n_each),
						len(allowed_combinations[i])
					)
					groups.iloc[compare_idx] = grp_indicators
					max_grp = np.max(grp_indicators) + 1
					break_search = True
					break
			if break_search:
				break
	return groups

def rm_false_positives(x):
	""" Remove false positive matches of gleason scores in text.
	Especially names of fields in text such as "gleason 6 or less" caused false positives.

	Args:
		x (str): text

	Returns:
		str: trimmed text
	"""
	rm = [
		ger.base_gleason_regex + "[ ]?4[ ](ja|tai|or|och|eller)[ ]5",
		"fokaalinen syöpä \\([^)]*\\)",
		"\\(gleason score 6 tai alle\\)"
	]
	for pat in rm:
		x = re.sub(pat,"", x) 
	return x

def prepare_text(x):
	"""Does everything needed to prepare text for the actual extraction; 
	i.e. normalises the text and removes false positives.
	
	Args:
		x (str): text

	Returns:
		str: prepared text
	"""
	x = rm_false_positives(normalise_text(x))
	# to remove certain expressions we know in advance to have no bearing
	# on gleason scores --- to shorten and simplify the text.
	x = re.sub("\\([^0-9]+\\)", " ", x) # e.g. "(some words here)"
	x = re.sub("\\([ ]*[0-9]+[ ]*%[ ]*\\)", " ", x) # e.g. "(45 %)"
	# remove a false positive. this should live in rm_false_positives!
	# e.g. "Is bad (Gleason score 9-10): no"
	re_field_name_gleason_range = "[(][ ]*" + ger.whitelist_gleason_word + "[^0-9]*" + "[5-9][ ]*[-][ ]*([6-9]|(10))" + "[ ]*[)]"
	x = re.sub(re_field_name_gleason_range, " ", x)
		
	return re.sub("[ ]+", " ", x)