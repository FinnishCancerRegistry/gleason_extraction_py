import os
# import re
import regex as re
import sys

import numpy as np
import pandas as pd
from numpy.core.fromnumeric import repeat

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import logs
logger = logs.logging.getLogger('utils')


def normalise_text(x):
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
	) -> pd.DataFrame:
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
		DataFrame: Table with (at least) columns
			`a`(Int64): primary Gleason score
			`b`(Int64): second Gleason score
			`t`(Int64): tertiary Gleason
			`c`Int64): Gleason scoresum
			`grp`(Int64): group, that tells which Gleason score elements belong together
	"""

	try:
		if dt.columns.isin(['grp','grp_type','type']).any():
			raise ValueError('Input dt should NOT have columns grp, grp_type or type, as these are added by this function into dt')
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
	dt = dt.copy()
	dt.reset_index(drop=True, inplace=True)

	dt["type"] = pd.Series(None, dtype = "string")
	for col_nm in ["a", "b", "t", "c"]:
		dt.loc[dt[col_nm].notnull(), "type"] = col_nm
	dt["grp"] = pd.Series(np.nan, dtype="Int64")

	n = dt.shape[0]
	max_grp = 0
	while (dt["grp"].isnull().any()):
		wh_first = int(dt.isnull()["grp"].idxmax())
		for i in range(len(allowed_combinations)):
			break_search = False
			for n_each in range(n_max_each):
				candidate = np.repeat(allowed_combinations[i], (n_each + 1))
				r = list(range(wh_first, (min(n, wh_first  + len(candidate) ))))				
				if (np.array_equal(np.array(dt.loc[r, "type"]), candidate)):
					dt.loc[r, "grp"] = max_grp
					dt.loc[r, "grp_type"] = "".join(candidate)
					max_grp = max_grp + 1
					break_search = True
					break
			if (break_search):
				break
	return dt
