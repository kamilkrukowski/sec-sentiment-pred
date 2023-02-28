import re
import sys
from bs4 import BeautifulSoup

import edgar.html as html
import edgar

DATA_DIR = 'data'

tikrs = ['aapl', 'msft', 'amzn', 'tsla', 'googl', 'goog',  'unh', 'jnj', 'cvx',
         'jpm', 'hd', 'v', 'pg']
metadata = edgar.Metadata(data_dir=DATA_DIR)

data = []
for tikr in tikrs:
    print(tikr)
    edgar.load_files(tikr, data_dir=DATA_DIR, document_type='8-K',
                     include_supplementary=True, force_remove_raw=True)

    submissions = metadata.get_submissions(tikr)
    for sub in submissions:
        files = edgar.get_files(tikrs=tikr, submissions=sub, metadata=metadata)
        out = []
        for file in files:
            # Skip unextracted files
            if not metadata._get_file(tikr, sub, file).get('extracted', False):
                continue;
            f = edgar.read_file(tikr, sub, file, document_type='8-K',
                                data_dir=DATA_DIR)
            out.append(f)
        attrs = metadata._get_submission(tikr, sub)['attrs']

        # Remove submissions without date-time
        if 'CONFORMED PERIOD OF REPORT' not in attrs:
            continue;

        date = attrs['CONFORMED PERIOD OF REPORT']
        data.append([tikr, sub, date, html.clean_text(
                        html.remove_tables(' '.join(out)))])


with open('8k_data.csv', 'w') as f:
    f.write(",TIKR,FORM_TYPE,submission,span_texts,Date\n")
    for idx, i in enumerate(data):
        f.write(f"{idx},{i[0]},8-K,{i[1]},{i[3]},{i[2]}\n")
