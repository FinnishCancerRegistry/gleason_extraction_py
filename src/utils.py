import os
import re
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


def determine_element_combinations(dt, n_max_each = 5):
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
			`a`(int/float): primary Gleason score
			`b`(int/float): second Gleason score
			`t`(int/float): tertiary Gleason
			`c`(int/float): Gleason scoresum
			`grp`(float): group, that tells which Gleason score elements belong together
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
	dt = dt.copy().reset_index(drop=True)
	def find_non_na_elem(elem_nm):
		return elem_nm.dropna(inplace = False).index[0]

	dt["type"] = dt[["a","b","t","c"]].apply(find_non_na_elem, axis=1)
	dt["grp"] = np.nan

	n = dt.shape[0]
	max_grp = 0


	while (dt["grp"].isnull().values.any()):		
		wh_first = dt.isnull()["grp"].idxmax()
		for i in range(len(allowed_combinations)):
			break_search = False
			for n_each in range(n_max_each):
				candidate = np.repeat(allowed_combinations[i], (n_each + 1))
				r = list(range(wh_first, (min(n, wh_first  + len(candidate) ))))				
				if (np.array_equal(np.array(dt["type"][r]), candidate)):
					dt.loc[r, "grp"] = max_grp
					dt.loc[r, "grp_type"] = "".join(candidate)
					max_grp = max_grp + 1
					break_search = True
					break
			if (break_search):
				break
	return dt



def typed_format_dt_to_standard_format_dt(dt):
	"""Reformat Gleason Score Components

	Combine individual Gleason components (A, B, T, C) into Gleason scores (A + B (+T) = C).

	Args:
		dt (DataFrame): table that will be manipulated. With (at least) columns
			`text_id`(int64): identifiers for individual texts; one text may have one
				or more rows in this table
			`obs_id`(int64): identifiers for observed components; must be unique to each row
			`match_type`(str): match type
			`a`(int/float): primary component value
			`b`(int/float): secondary component value
			`t`(int/float): tertiary component value
			`c`(int/float): Gleason scoresum
			`warning`(str): warning message
		int/float: non-nan values int, nan-valeus float

	Convert from so-called "typed" formatting (where we remember the type of the
	match, e.g. "A + B = C", "identifier-C", etc.) to "standard" formatting
	(where we forget the type of the match and combine individual primary and
	secondary Gleason components into at least A + B scores).

	This function relies on `text_id` to ensure components are not combined 
	across different texts and `obs_id` for the combinations themselves.
	`dt` is asserted to be keyed (sorted) by first `text_id` and then `obs_id`.
	At least one of the columns `a`, `b`,`t` and `c` should be non-NA.

	`obs_id` should contain running numbers. Whether components are attempted to 
	be combined depends on whether the components are "adjacent," i.e. whether 
	`obs_id[i-1] == obs_id[i] - 1`. When there are two or more such adjacent 
	components in sequence (e.g. `obs_id` values 1:5), these are temporarily 
	grouped and attempted to be combined. E.g. the input table for text 
 
 			"primary gleason 3, gleason 3 + 4, secondary gleason 4"
	
	has three rows with {A=3}, {A=3, B=4}, and {B=4}. Here row {A=3, B=4} is not
	considered for grouping, but neither are {A=3} and {B=4}, because {A=3, B=4} 
	is between them, breaking the adjacency.
	
	Combinations are identified using function `determine_element_combinations`. It
	is run separately for each temporary group based on `obs_id` adjacency.
 
	Any "orphan" A or B values are retained. E.g. if {A,B,A} are associated with
	one text, the three values are reformatted into {{A,B}, A}.

	Raises:
		ValueError: obs_id need to be unique
		ValueError: Match type need to be among "a", "b", "c", "t", "a + b", "a + b = c", "a + b + t = c", "a + b + t", "kw_all_a", nan
		ValueError: There has to be at least one non-nan a/b/t/c value per row


	Returns:
		`DataFrame`: Table with all the same columns as Args `dt` except `match_type`

		The rows are ordered by `text_id` and `obs_id`
	"""

	logger.info('Combine individual gleason components into gleason scores')
	out = pd.DataFrame(columns=["text_id", "obs_id", "a", "b", "t","c"])

	try:
		if not dt.obs_id.is_unique:
			print(dt[~dt.obs_id.duplicated(keep=False)])
			raise ValueError('obs_id need to be unique')
		if not dt.match_type.isin(["a", "b", "c", "t", "a + b", "a + b = c", "a + b + t = c", "a + b + t", "kw_all_a", np.nan]).all():
			raise ValueError('Match type need to be among "a", "b", "c", "t", "a + b", "a + b = c", "a + b + t = c", "a + b + t", "kw_all_a", nan')
		if not dt.loc[dt[['a','b','t','c']].notnull().sum(axis=1) == 0].empty:
			raise ValueError('There has to be at least one non-nan a/b/t/c value per row')
		# if not dt.a.isin([2,3,4,5, np.nan]).all():
		# 	raise ValueError('Gleason A should be among 2,3,4,5,nan')
		# if not dt.b.isin([2,3,4,5, np.nan]).all():
		# 	raise ValueError('Gleason B should be among 2,3,4,5,nan')
		# if not dt.t.isin([2,3,4,5, np.nan]).all():
		# 	raise ValueError('Tertiary gleason should be among 2,3,4,5,nan')
		# if not dt.c.isin([4,5,6,7,8,9,10,np.nan]).all():
		# 	raise ValueError('Gleason C should be among 4,5,6,7,8,9,10, nan')
	except Exception as e:
		logger.exception(e)
		raise

	dt = dt.copy()
	is_single_elem_match = dt[["match_type"]].isin(["a", "b", "t", "c"])
	elem_dt = dt[is_single_elem_match["match_type"] == True].copy()
	elem_dt = elem_dt.sort_values(by=['text_id', 'obs_id']).reset_index(drop = True) # not necessarily needed?

	if (len(elem_dt) > 0):
		# sequential observations have diff(obs_id) == 1, non-seq. have > ;
		# latter cases are marked as the first observations in their own group of sequential observations
		wh_first_in_seq_set = np.append(np.array([0]), elem_dt.where(elem_dt["obs_id"].diff().to_frame() > 1)["obs_id"].dropna().index.to_numpy())
		wh_last_in_seq_set = np.append(wh_first_in_seq_set -1, elem_dt.shape[0] - 1)
		wh_last_in_seq_set = wh_last_in_seq_set[wh_last_in_seq_set >= 0]
		elem_dt[".__processing_grp"] = np.nan
		# create temporary groups of observations
		for l in range(len(wh_first_in_seq_set)):
			elem_dt.loc[(elem_dt.index >= wh_first_in_seq_set[l]) & (elem_dt.index <= wh_last_in_seq_set[l]), ".__processing_grp"] = l
		elem_dt[".__processing_grp"] = elem_dt.groupby(["text_id", ".__processing_grp"]).ngroup()
		elem_dt = elem_dt.groupby(".__processing_grp", group_keys = False).apply(lambda x : determine_element_combinations(x, n_max_each = 6))
		elem_dt[".__processing_grp"] = elem_dt.groupby(["grp", ".__processing_grp"]).ngroup()

		def combine_groups(dt):
			sub_dt = dt.copy().reset_index(drop=True)
			a  = sub_dt[sub_dt["a"].notnull()]["a"].reset_index(drop=True)
			b  = sub_dt[sub_dt["b"].notnull()]["b"].reset_index(drop=True)
			t  = sub_dt[sub_dt["t"].notnull()]["t"].reset_index(drop=True)
			c  = sub_dt[sub_dt["c"].notnull()]["c"].reset_index(drop=True)
			n = max(len(a), len(b), len(t), len(c))
			if (len(a) == 0):
				a = pd.DataFrame(repeat(np.nan, n))
			if (len(b) == 0):
				b = pd.DataFrame(repeat(np.nan, n))
			if (len(t) == 0):
				t = pd.DataFrame(repeat(np.nan, n))
			if (len(c) == 0):
				c =  pd.DataFrame(repeat(np.nan, n))
			values = pd.concat([a,b,t, c], axis = 1)
			values.columns = ["a", "b", "t", "c"]
			sub_dt = sub_dt[0:n]
			sub_dt[['a','b','t','c']] = values
			return sub_dt
		elem_dt = elem_dt.groupby(".__processing_grp", group_keys = False).apply(combine_groups)
		elem_dt = elem_dt.sort_values(['text_id','obs_id'], ignore_index = True)
	out = pd.concat([elem_dt, dt[~is_single_elem_match.match_type]], ignore_index = True)
	out = out[['text_id', 'obs_id', 'a', 'b', 't', 'c', 'warning']]
	out = out.sort_values(by=['text_id','obs_id'], ignore_index=True) # order by text_id and obs_id
	return out


# input_dt_example = pd.DataFrame({
# 	"text_id": [1,1,2,2,3],
# 	"obs_id": [1,2,3,4,5],
# 	"match_type": ['a', 'b', 'a', 'b', 'a + b = c'],
# 	"a": [4, np.nan, 5, np.nan, 4],
# 	"b": [np.nan, 3, np.nan, 4, 4],
# 	"c": [np.nan, np.nan, np.nan, np.nan, 8]		
# })
# print(typed_format_dt_to_standard_format_dt(input_dt_example))
