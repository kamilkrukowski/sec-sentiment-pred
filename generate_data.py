import sys

import edgar.html as html
import edgar

DATA_DIR = 'data'
DELETE_RAW = True

tikrs = ['aapl', 'msft', 'amzn', 'tsla', 'googl', 'goog',  'unh', 'jnj', 'cvx',
         'jpm', 'hd', 'v', 'pg']
metadata = edgar.Metadata(data_dir=DATA_DIR)

#  Prepare file for constant offloading
fout = open('8k_data.csv', 'w');
fout.write(",TIKR,FORM_TYPE,submission,span_texts,Date\n")
frowidx = 0


data = []
for tikr in tikrs:
    print(tikr)
    edgar.load_files(tikr, data_dir=DATA_DIR, document_type='8-K',
                     include_supplementary=True, force_remove_raw=DELETE_RAW)

    submissions = metadata.get_submissions(tikr)
    for sub in submissions:
        files = edgar.get_files(tikrs=tikr, submissions=sub, metadata=metadata)

        # Store consecutive supplementary file contents as text stream to
        # concatenate later.
        out = []
        for file in files:
            # Skip unextracted files
            if not metadata._get_file(tikr, sub, file).get('extracted', False):
                continue;

            f = edgar.read_file(tikr, sub, file, document_type='8-K',
                                data_dir=DATA_DIR)
            out.append(f)

        entry = html.clean_text(html.remove_tables(' '.join(out)))

        attrs = metadata._get_submission(tikr, sub)['attrs']
        # Remove submissions without date-time
        if 'CONFORMED PERIOD OF REPORT' not in attrs:
            continue;
        date = attrs['CONFORMED PERIOD OF REPORT']

        # Write to file
        fout.write(f"{frowidx},{tikr},8-K,{sub},{entry},{date}\n")
        frowidx += 1

#  Close data pipe file
fout.close()
