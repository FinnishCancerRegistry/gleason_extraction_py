import regex as re
import pandas as pd
import typing as ty
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import utils as geut
import gleason_extraction_regexes as ger
import logs
logger = logs.logging.getLogger('gleason_extraction') # type: ignore

# extraction funs ---------------------------------------------------------

def extract_gleason_scores(
		texts : ty.Iterable[str],
		text_ids : ty.Iterable[int],
		patterns : ty.Iterable[str | re.Pattern] = ger.fcr_pattern_dt()["full_pattern"],
		match_types : None | ty.Iterable[str] = ger.fcr_pattern_dt()["match_type"]
	):
	"""Extract Gleason Scores.

		Runs the extraction itself, parses and formats results.

	Args:
		`texts` (ty.Iterable[str]):
			Texts to process
		`text_ids` (ty.Iterable[int]):
			Identifies each text; will be retained in output
		`patterns` (ty.Iterable[str | re.Pattern]):
			Each element is passed to `regex.compile`.
		`match_types` (None | ty.Iterable[str]):
			If not `None`, must be same length as `patterns`. Used to create column
			`match_type` in output. Can be useful for debugging.

	Raises:
		ValueError: Number of texts and text ids have to match
		ValueError: text ids must be unique

	Returns:
		`DataFrame`: Gleason extraction table with columns
			`text_id` (Int64): original text_id
			`obs_id` (Int64): original text_id extended with the value, 
				which tells the order of appearance in the text. E.g for text_id 1, obs_id is 1001
			`a` (Int64): Most prevalent (primary) grade value
			`b` (Int64): Second most prevalent (secondary) grade value
			`t` (Int64): Third most common (tertiary) grade value
			`c` (Int64): Gleason scoresum
			`match_type` (str|None):
				If `match_types` is passed, this is for row `i` the `match_types[j]`
				when the `j`th pattern in `patterns` was used to extract the values
				in this row. If more than one pattern was used, the
				`match_types[j]`, `match_types[k]`, etc. are concatenated into one
				string via `"; ".join(x)`.
			`warning` (str|None):
				May contain a warning if the `match_types` element of the correspoding
				`patterns` element that was used to extract the values in the row
				do not match with what was actually extracted. For instance if for some
				reason the `match_types` element was `"a + b = c"` but only `a` and `b`
				were extracted then this column will contain a warning about that.
				More than one warning are concatenated into one string via
				`"; ".join(x)`. I think this is these days always `None`.
			  
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
		"stop": [],
		"match_type": []
	}

	value_type_space = ["A", "B", "T", "C"]
	compiled_patterns = list(map(re.compile, patterns))
	if match_types is None:
		match_types : list[None | str] = [None] * len(compiled_patterns)
	for text, text_id in zip(texts, text_ids):
		assert isinstance(text, str),\
			"The text with `text_id = %i` was not a string but of type `%s`"\
				% (text_id, str(type(text)))
		text = geut.prepare_text(text)
		for cp, cp_type in zip(compiled_patterns, match_types):
			for m in cp.finditer(text):
				ms = m.span()
				text = text[:ms[0]] + "_" * (ms[1] - ms[0]) + text[ms[1]:]
				cd = m.capturesdict()
				if "A_and_B" in cd.keys() and not cd["A_and_B"] is None:
					# this handles regexes which capture e.g.
					# "sample was entirely grade 3", meaning 3 + 3.
					# from the example above, 3 needs to be extracted by named capture
					# group `A_and_B`.
					cd = {
						"A": cd["A_and_B"],
						"B": cd["A_and_B"]
					}
				# e.g. cd = {"A": [3, 3], "B": [3, 4]} when text was
				# "gleason 3 + 3 / 3 + 4"
				value_type_set = list(np.intersect1d(value_type_space, list(cd.keys())))
				for elem_nm in value_type_set:
					for value in cd[elem_nm]:
						out[elem_nm.lower()].append(int(value))
				# now e.g. [3, 3] has been appended to out["a"] and [3, 4] to out["b"].
				# then the other columns.
				n_add = len(out[value_type_set[0].lower()]) - len(out["text_id"])
				for _ in range(n_add):
					null_value_col_nm_set = [
						str(x).lower() for x in list(np.setdiff1d(
							value_type_space, value_type_set
						))
					]
					for null_value_col_nm in null_value_col_nm_set:
						# pad with None those values which were not extracted. e.g. if text
						# was "gleason 3 + 3 / 3 + 4" we append None twice to
						# out["c"] because of `for add_row_no in range(n_add):`
						out[null_value_col_nm.lower()].append(None)
					# the same thing is added as many times as necessary to have the same
					# length in every column. note that e.g. "gleason 3 + 3 / 3 + 4"
					# produces two rows where both rows have the same `start` and `stop`
					# because the same regex captures both alternatives.
					out["start"].append(ms[0])
					out["stop"].append(ms[1])
					out["text_id"].append(text_id)
					out["obs_id"].append(None)
					out["match_type"].append(cp_type)
	value_col_nm_space = [x.lower() for x in value_type_space]
	for int_col_nm in ["text_id", "obs_id"] + value_col_nm_space:
		out[int_col_nm] = pd.Series(out[int_col_nm], dtype="Int64")
	out : pd.DataFrame = pd.DataFrame(out)
	out.sort_values(by=["text_id", "start", "stop"], inplace=True)
	out.reset_index(drop=True, inplace=True)
	out["warning"] = geut.make_column_warning(
		match_types=out["match_type"],
		a=out["a"],
		b=out["b"],
		t=out["t"],
		c=out["c"]
	)

	logger.info('extract_gleason_scores starts combining any orpshan values')
	is_orphan = out.loc[:, value_col_nm_space].notnull().sum(axis=1) == 1
	if is_orphan.sum() > 0:
		out_list : list[pd.DataFrame] = [out.loc[~is_orphan, :]]
		out_orphan = out.loc[is_orphan, :]
		text_id_set = out_orphan["text_id"].unique()
		for i in range(len(text_id_set)):
			is_text_id = out_orphan["text_id"] == text_id_set[i]
			tmp_df = out_orphan.loc[is_text_id, :].copy()
			tmp_df.reset_index(inplace=True, drop=True)
			if is_text_id.sum() > 1:
				tmp_df["grp"] = geut.determine_element_combinations(dt = tmp_df)
				# initially tmp_df contains e.g.
        # a = [4, np.nan], b = [np.nan, 3], start = [8, 27], stop = [19, 39]
				tmp_df = tmp_df.groupby(["text_id", "grp"])[
					["a", "b", "t", "c", "start", "stop", "match_type", "warning"]
				].aggregate(
					func={
						"a": geut.aggregate_column_a_b_t_c,
						"b": geut.aggregate_column_a_b_t_c,
						"t": geut.aggregate_column_a_b_t_c,
						"c": geut.aggregate_column_a_b_t_c,
						"start": "min",
						"stop": "max",
						"match_type": geut.aggregate_column_match_type,
						"warning": geut.aggregate_column_match_type
					}
				)
				# ok now tmp_df is e.g.
        # a = [4], b = [3], start = [8], stop = [39]
				tmp_df.reset_index(drop=False, inplace=True)
			out_list.append(tmp_df)
		out = pd.concat(out_list)
	out["obs_id"] = out.groupby("text_id")["text_id"].apply(
			lambda x: pd.Series(np.arange(len(x)))
		).reset_index(drop=True)
	out.sort_values(by=["text_id", "obs_id"], inplace=True)
	out.reset_index(inplace=True, drop=True)
	out["warning"] = pd.Series(None, dtype="string")
	if "grp" in out.columns:
		out.drop(columns="grp")
	logger.info('extract_gleason_scores finished')
	return out
