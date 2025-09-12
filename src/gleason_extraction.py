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

def extract_gleason_scores_from_texts(
		texts : ty.Iterable[None | str],
		text_ids : None | ty.Iterable[int] = None,
		patterns : ty.Iterable[str | re.Pattern] = ger.fcr_pattern_dt()["full_pattern"],
		match_types : None | ty.Iterable[str] = ger.fcr_pattern_dt()["match_type"]
	):
	"""Extract Gleason Scores from Texts.

	Runs `extract_gleason_scores_from_text` for each element of `texts`.

	Args:
		`texts` (ty.Iterable[None | str]):
			Texts to process.
		`text_ids` (None | ty.Iterable[int]):
			Identifier for each text. If `None`, a running number starting from zero
			is used.

	Returns:
		`pd.DataFrame`. It has all the columns that the output of
		`extract_gleason_scores_from_text` has plus column `text_id`.
	"""

	logger.info('extract_gleason_scores_from_texts called')
	try:
		if not isinstance(texts, ty.Iterable):
			raise TypeError("Arg `texts` must be Iterable")
		if not isinstance(text_ids, (ty.Iterable, type(None))):
			raise TypeError("Arg `text_ids` must be Iterable")
	except Exception as e:
		logger.exception(e)
		raise
	if text_ids is None:
		iterator = enumerate(texts)
	else:
		iterator = zip(text_ids, texts)
	out = []
	for text_id, text in iterator: # type: ignore
		# these two for pylance
		text : str | None = text
		text_id : int  = text_id
		df = extract_gleason_scores_from_text(
			text=text,
			patterns=patterns,
			match_types=match_types
		)
		df.insert(
			column="text_id",
			loc=0,
			value=pd.Series(np.repeat(text_id, df.shape[0]), dtype="Int64"),
			allow_duplicates=True
		)
		out.append(df)
	out = pd.concat(out)
	out.reset_index(inplace=True, drop=False)
	logger.info('extract_gleason_scores_from_texts finished')
	return out

def extract_gleason_scores_from_text(
		text : None | str,
		patterns : ty.Iterable[str | re.Pattern] = ger.fcr_pattern_dt()["full_pattern"],
		match_types : None | ty.Iterable[str] = ger.fcr_pattern_dt()["match_type"] # type: ignore
	):
	"""Extract Gleason Scores from a Text.

		Runs the extraction itself, parses and formats results.

	Args:
		`text` (None | str):
			Text to process.
		`patterns` (ty.Iterable[str | re.Pattern]):
			Each element is passed to `regex.compile`.
		`match_types` (None | ty.Iterable[str]):
			If not `None`, must be same length as `patterns`. Used to create column
			`match_type` in output. Can be useful for debugging.

	Returns:
		`DataFrame`: Gleason extraction table with columns.
			`obs_id` (Int64):
				Running number starting from zero that identifies the observation.
			`a` (Int64):
				Most prevalent (primary) grade value.
			`b` (Int64):
				Second most prevalent (secondary) grade value.
			`t` (Int64):
				Third most common (tertiary) grade value.
			`c` (Int64):
				Gleason scoresum.
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
	assert isinstance(text, (str, type(None))),\
		"Arg `text` was not `str` nor `None` but %s" % str(type(text))
	
	out : dict[str, list[int | str | None] | pd.Series] = { # type: ignore
		"obs_id": [],
		"a": [],
		"b": [],
		"t": [],
		"c": [],
		"start": [],
		"stop": [],
		"match_type": [],
		"warning": []
	}
	out_dtype_dict = {
		"obs_id": "Int64",
		"a": "Int64",
		"b": "Int64",
		"t": "Int64",
		"c": "Int64",
		"start": "Int64",
		"stop": "Int64",
		"match_type": "str",
		"warning": "str"
	}
	if text is None:
		for col_nm in out_dtype_dict:
			out[col_nm] = pd.Series(out[col_nm], dtype=out_dtype_dict[col_nm])
		return pd.DataFrame(out)

	value_type_space = ["A", "B", "T", "C"]
	compiled_patterns = list(map(re.compile, patterns))
	if match_types is None:
		match_types : list[None | str] = [None] * len(compiled_patterns)
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
			n_add = len(out[value_type_set[0].lower()]) - len(out["start"])
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
				out["obs_id"].append(None)
				out["warning"].append(None)
				out["match_type"].append(cp_type)
	value_col_nm_space = [x.lower() for x in value_type_space]
	for col_nm in out_dtype_dict:
		out[col_nm] = pd.Series(out[col_nm], dtype=out_dtype_dict[col_nm])
	out : pd.DataFrame = pd.DataFrame(out)
	out.sort_values(by=["start", "stop"], inplace=True)
	out.reset_index(drop=True, inplace=True)
	out["warning"] = geut.make_column_warning(
		match_types=out["match_type"],
		a=out["a"],
		b=out["b"],
		t=out["t"],
		c=out["c"]
	)

	is_orphan = out.loc[:, value_col_nm_space].notnull().sum(axis=1) == 1
	if is_orphan.sum() > 1:
		nonorphan_df = out.loc[~is_orphan, :]
		orphan_df = out.loc[is_orphan, :].copy()
		orphan_df.reset_index(inplace=True, drop=True)
		orphan_df["grp"] = geut.determine_element_combinations(dt = orphan_df)
		# initially tmp_df contains e.g.
		# a = [4, np.nan], b = [np.nan, 3], start = [8, 27], stop = [19, 39]
		orphan_df = orphan_df.groupby("grp")[
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
		orphan_df.reset_index(drop=False, inplace=True)
		# ok now tmp_df is e.g.
		# a = [4], b = [3], start = [8], stop = [39]
		out = pd.concat([nonorphan_df, orphan_df])
	
	out.sort_values(by=["start", "stop"], inplace=True)
	out.reset_index(inplace=True, drop=True)
	out["obs_id"] = np.arange(out.shape[0])
	# correct column order and drop extra columns
	out = out[list(out_dtype_dict.keys())]
	return out
