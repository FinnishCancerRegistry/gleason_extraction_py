import regex as re
import pandas as pd
import typing as ty
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import utils as ut
import gleason_extraction_regexes as ger
import logs
logger = logs.logging.getLogger('gleason_extraction') # type: ignore

# extraction funs ---------------------------------------------------------
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
	x = rm_false_positives(ut.normalise_text(x))
	# to remove certain expressions we know in advance to have no bearing
	# on gleason scores --- to shorten and simplify the text.
	x = re.sub("\\([^0-9]+\\)", " ", x) # e.g. "(some words here)"
	x = re.sub("\\([ ]*[0-9]+[ ]*%[ ]*\\)", " ", x) # e.g. "(45 %)"
	# remove a false positive. this should live in rm_false_positives!
	# e.g. "Is bad (Gleason score 9-10): no"
	re_field_name_gleason_range = "[(][ ]*" + ger.whitelist_gleason_word + "[^0-9]*" + "[5-9][ ]*[-][ ]*([6-9]|(10))" + "[ ]*[)]"
	x = re.sub(re_field_name_gleason_range, " ", x)
		
	return re.sub("[ ]+", " ", x)

def extract_gleason_scores(
		texts : ty.Iterable[str],
		text_ids : ty.Iterable[int],
		pattern_dt : pd.DataFrame = ger.fcr_pattern_dt()
	):
	"""Extract Gleason Scores.

		Runs the extraction itself, parses and formats results.

	Args:
		`texts` (ty.Iterable[str]): texts to process
		`text_ids` (ty.Iterable[int]): identifies each text; will be retained in output
		`pattern_dt` (DataFrame, optional): Defaults to fcr_pattern_dt(). With columns
			`pattern_name` (str): one name per pattern
			`match_type` (str): value combination to look for
			`prefix` (str): regex; context prefix for value
			`value` (str): regex; the value itself
			`suffix` (str): regex; context suffix for value
			`full_pattern` (str): regex; prefix + value + suffix

	Raises:
		ValueError: Number of texts and text ids have to match
		ValueError: Format needs to be either "standard" or "typed"
		ValueError: Each match type in pattern_dt have to exist in instructions dict as well
		ValueError: text ids must be unique

	Returns:
		`DataFrame`: Gleason extraction table with columns
			`text_id` (Int64): original text_id
			`obs_id` (Int64): original text_id extended with the value, 
				which tells the order of appearance in the text. E.g for text_id 1, obs_id is 1001
			`a` (Int64): Most prevalent gleason value
			`b` (Int64): Second most prevalent gleason value
			`t` (Int64): Tertiary gleason value
			`c` (Int64): Gleason score
			`warning` (str): Warning message (match type does not match extracted values, gleason score does not match primary and secondary values)
	In `standard` format, gleason elements are combined so that possible abc values of one component are at the same row.
	Else output contains columns `combination_id`,`value_type`, `value` to link
	values to combinations to texts.
	Non-nan values are int and nan values None.
	"""

	logger.info('extract_gleason_scores called')
	try:
		if not isinstance(texts, ty.Iterable):
			raise TypeError("Arg `texts` must be Iterable")
		if not isinstance(text_ids, ty.Iterable):
			raise TypeError("Arg `text_ids` must be Iterable")
		n_text_ids = len(list(text_ids))
		n_texts = len(list(texts))
		if not (len(set(text_ids)) == n_text_ids):
			raise ValueError("Arg `text_ids` must not contain duplicated values")
		if not (n_texts == n_text_ids):
			raise ValueError('Args `texts` and `text_ids` must be of same length')
	except Exception as e:
		logger.exception(e)
		raise

	logger.info('extract_gleason_scores starts loop over regexes')
	out : dict[str, list[int | str | None] | pd.Series] = { # type: ignore
		"text_id": [],
		"obs_id": [],
		"a": [],
		"b": [],
		"t": [],
		"c": [],
		"start": [],
		"stop": []
	}

	patterns = list(map(re.compile, pattern_dt["full_pattern"]))
	i = 0
	for text, text_id in zip(texts, text_ids):
		i += 1
		assert isinstance(text, str),\
			"The %i'th text was not a string but of type `%s`" % (i, str(type(text)))
		n_extractions = 0
		text = prepare_text(text)
		for pattern_no in range(len(patterns)):
			for m in patterns[pattern_no].finditer(text):
				n_extractions += 1
				out["text_id"].append(text_id)
				gd = m.groupdict()
				for elem_nm in ["A", "B", "T", "C"]:
					if elem_nm in gd:
						out[elem_nm.lower()].append(int(gd[elem_nm]))
					else:
						out[elem_nm.lower()].append(None)
				ms = m.span()
				out["start"].append(ms[0])
				out["stop"].append(ms[1])
				text = text[:ms[0]] + "_" * (ms[1] - ms[0]) + text[ms[1]:]
	for int_col_nm in ["text_id", "obs_id", "a", "b", "t", "c"]:
		out[int_col_nm] = pd.Series(out[int_col_nm], dtype="Int64")
	out : pd.DataFrame = pd.DataFrame(out) # type: ignore
	out.sort_values(by=["text_id", "start"], inplace=True)
	out.reset_index(drop=True, inplace=True)

	logger.info('extract_gleason_scores starts combining any oprhan values')
	is_orphan = out.loc[:, ['a','b','t','c']].notnull().sum(axis=1) == 1
	if is_orphan.sum() > 0:
		out_list : list[pd.DataFrame] = [out.loc[~is_orphan, :]]
		out_orphan = out.loc[is_orphan, :]
		text_id_set = out_orphan["text_id"].unique()
		for i in range(len(text_id_set)):
			is_text_id = out_orphan["text_id"] == text_id_set[i]
			tmp_df = out_orphan.loc[is_text_id, :]
			if is_text_id.sum() > 1:
				tmp_df = ut.determine_element_combinations(dt = tmp_df)
				# initially tmp_df contains e.g.
        # a = [4, np.nan], b = [np.nan, 3], start = [8, 27], stop = [19, 39]
				tmp_df = tmp_df.melt(id_vars=["text_id", "grp"], value_vars=["a", "b", "t", "c", "start", "stop"])
				# fist i.e. lowest start value kept, and highest stop value
				tmp_df.loc[np.bitwise_and(tmp_df.duplicated(subset=["grp", "variable"]), tmp_df["variable"] == "start"), "value"] = np.nan
				tmp_df.loc[np.bitwise_and(tmp_df.duplicated(subset=["grp", "variable"], keep="last"), tmp_df["variable"] == "stop"), "value"] = np.nan
				tmp_df = tmp_df.loc[tmp_df["value"].notnull(), :]
				# lambda just to keep the original dtype. default mean produces floats.
				tmp_df = tmp_df.pivot_table(index = ["text_id", "grp"], columns="variable", values="value", aggfunc=lambda x: x)
				tmp_df.reset_index(drop=False, inplace=True)
				# ok now tmp_df is e.g.
        # a = [4], b = [3], start = [8], stop = [39]
			out_list.append(tmp_df)
		out = pd.concat(out_list)
	out["obs_id"] = 1000 * out["text_id"] +\
		out["text_id"].duplicated(keep="first").cumsum()
	out.sort_values(by=["text_id", "obs_id"], inplace=True)
	out.reset_index(inplace=True, drop=True)
	out["warning"] = pd.Series(None, dtype="string")
	logger.info('extract_gleason_scores finished')
	return out
