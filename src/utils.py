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

__regex_expected_value_dict__ = {
	"a + b = c" : ["a", "b", "c"],
	"a + b" : ["a", "b"],
	"a + b + t = c" : ["a", "b", "t", "c"],
	"a + b + t" : ["a", "b", "t"],
	"t" : ["t"],
	"b" : ["b"],
	"a" : ["a"],
	"c" : ["c"],
	"kw_all_a" : ["a", "b"]
}

def make_warning(
	match_type : None | pd.api.typing.NAType | str,
	a : None | pd.api.typing.NAType | np.integer | int,
	b : None | pd.api.typing.NAType | np.integer | int,
	t : None | pd.api.typing.NAType | np.integer | int,
	c : None | pd.api.typing.NAType | np.integer | int
) -> str | None:
	if match_type is None or pd.isna(match_type) or\
		not match_type in __regex_expected_value_dict__.keys():
		return None
	
	required_value_nms : list[str] = __regex_expected_value_dict__[match_type]
	value_dict = {
		"a": a,
		"b": b,
		"t": t,
		"c": c
	}
		
	w = []
	missing_required_value_nms = []
	for rvn in required_value_nms:
		if pd.isna(value_dict[rvn]):
			missing_required_value_nms.append(rvn)
	if len(missing_required_value_nms) > 0:
		w.append(
			"Pattern was supposed to extract `%s` but did not extract %s" %\
				(match_type, ", ".join(missing_required_value_nms))
		)

	if match_type in ["a + b = c", "a + b + t = c"] and\
		not pd.isna(a) and not pd.isna(b) and not pd.isna(c) and\
		a + b != c:
			w.append("Extracted a + b != c")

	if len(w) == 0:
		w = None
	else:
		w = "; ".join(w)
	
	return w

import typing as ty
def make_column_warning(
	match_types : None | ty.Iterable[str],
	a : ty.Iterable[None | pd.api.typing.NAType | np.integer | int],
	b : ty.Iterable[None | pd.api.typing.NAType | np.integer | int],
	t : ty.Iterable[None | pd.api.typing.NAType | np.integer | int],
	c : ty.Iterable[None | pd.api.typing.NAType | np.integer | int]
) -> pd.Series:
	w = []
	if match_types is None:
		if isinstance(a, (list, pd.Series, np.ndarray)):
			n = len(a)
		else:
			n = 0
			for a_value in a:
				n += 1
			w : list[None | str] = [None] * n
		return pd.Series(w, dtype="str")
	
	for mt_value, a_value, b_value, t_value, c_value in zip(match_types, a, b, t, c):
		w.append(make_warning(
			match_type=mt_value,
			a=a_value,
			b=b_value,
			t=t_value,
			c=c_value
		))
	
	return(pd.Series(w, dtype="str"))

def aggregate_column_warning(warning : pd.Series) -> str | None:
	is_missing = warning.isnull()
	if is_missing.all():
		return(None)
	return "; ".join(warning[~is_missing])

def aggregate_column_match_type(match_type : pd.Series) -> str | None:
	is_missing = match_type.isnull()
	if is_missing.all():
		return(None)
	return "; ".join(match_type[~is_missing])

def aggregate_column_a_b_t_c(value_column : pd.Series) -> int | None:
	is_missing = value_column.isnull()
	not_missing = ~is_missing
	if is_missing.all():
		return(None)
	elif not_missing.sum() > 1:
		raise ValueError(
			"Internal error: Multiple non-na values when combining groups."
		)
	return value_column[not_missing].iloc[0]
