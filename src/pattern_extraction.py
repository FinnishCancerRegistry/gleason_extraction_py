import os
import re
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import logs

logger = logs.logging.getLogger('pattern_extraction')


def extract_context_affixed_values(text, pattern_dt, mask = None, n_max_tries_per_pattern = 100):
	"""Extract substrings (values) from text with context prefixes and suffixes.

	Args:
			`text` (list(str)): texts to extract values from
			`pattern_dt` (DataFrame): with columns
				`pattern_name` (str): one name per pattern; these will be columns in output
				`match_type` (str): not used here
				`prefix` (str): context prefix for value
				`value` (str): the value itself
				`suffix` (str): context suffix for value
			`mask` (str): optional. Each time a match is found in an element of `text`, the match is replaced by a mask to avoid matching the same thing multiple times; 
				Default mask: `____________________%PATTERN_NAME%:%ORDER%____________________`
			`n_max_tries_per_pattern`(int): defaults to 100
		
	
	If a value is found n times in a `text` element, there will be n values
	for that element in output (all matches are extracted for each pattern).
	
	The patterns are processed in the given order. Hence you may have even e.g.
	a special case of another pattern and process that first to ensure more
	exact matching. After extracting a pattern (defined by `prefix`, `value`,
	and `suffix` pasted together), that match is replaced in `text` with
	 `mask` to ensure that once a match is found by a pattern,
	 consequent patterns cannot match to the same part of the text.
 
	If an element of `text` has no matches in any of the given patterns, 
	there will be zero rows in output for that element.


	Returns:
			`DataFrame` with columns
				`pos` (int): 	index of input text. Each element identifies which element of text the value was extracted from
				`pattern_name` (str): identifying the pattern used to extract the corresponding value
				`value` (str): extracted values (with any context stripped)

				The rows are in the same order as `text` and, within an element of `text`, in the order of appearance.		
	"""

	
	if not mask:
		mask = "_" * 20 + "%PATTERN_NAME%:%ORDER%" + "_" * 20

	try:
		if not bool(re.search("%ORDER%", mask)):
			raise ValueError('Mask was {}. It should contain "%ORDER%".'.format(mask))
	except Exception as e:
		logger.exception(e)
		raise

	pattern_dt.loc[:, "full_pattern"] = "(?P<prefix>" + pattern_dt["prefix"] + ")" + "(?P<value>" + pattern_dt["value"] + ")" + "(?P<suffix>" + pattern_dt["suffix"] + ")"	

	#init return value
	extr_dt = pd.DataFrame(columns = ['pos', 'pattern_name', 'value']) 

	#start extracting...
	for idx, elem in enumerate(text):
		extracted = []
		pattern_names = []
		pos = []
		text_elem = elem

		if (text_elem):
			#look for each pattern one by one
			for index, row in pattern_dt.iterrows():    
				pattern_name = row["pattern_name"]
				pattern = row['full_pattern']
				n_tries = 0

				#iterate the text element since one pattern may appear many times
				while (n_tries < n_max_tries_per_pattern and re.search(pattern, text_elem)):
					n_tries = n_tries + 1
					newly_extracted = re.search(pattern, text_elem).group('value')
					extracted.append(newly_extracted)
		
					pattern_names.append(pattern_name)
					pos.append(idx)					
					try:
						if not (len(extracted) < 1000):
							raise ValueError('Looks like you had at least 1000 matches in string which is not supported. {}'.format(text_elem))
					except Exception as e:
						logger.exception(e)
						raise
					mask_num = str(len(extracted) - 1).zfill(3)
					new_mask = re.sub("%ORDER%", ("%ORDER=" + mask_num + "%"), mask)
					new_mask = re.sub("%PATTERN_NAME%", ("%PATTERN_NAME=" + pattern_name + "%"), new_mask)
					text_elem = re.sub(pattern, new_mask, text_elem, 1) # mask the first occurence

			#match(es) found, now order them
			if (len(extracted) > 0):
				match_order = re.findall("%ORDER=[0-9]+%", text_elem)
				order_in_text = [int(re.search("[0-9]+", m).group(0)) for m in match_order]     
				dt = pd.DataFrame({'pos': pos , 'pattern_name': pattern_names, 'value': extracted}) # in the order of they were found
				dt = dt.loc[order_in_text].reset_index(drop = True) # in the order of appearance
				extr_dt = pd.concat([extr_dt, dt], ignore_index = True)
	return extr_dt