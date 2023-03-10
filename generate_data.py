from tqdm.auto import tqdm

import edgar
from parser_8k import Parser_8K

DATA_DIR = 'data'
# Whether to include type of 8-K document in embedding.
USE_DOC_SUBTYPES = True

with open('tickrs.txt') as f:  # tikr list right here
    l = f.read()
tikrs = [i.split(',')[0] for i in l.split('\n')]

#  Prepare file for constant offloading
fout = open('8k_data.tsv', 'w', encoding='utf-8')
fout.write("\ttikr\tFORM_TYPE\tsubmission\ttext\tDate\n")
frowidx = 0

config = edgar.DataLoaderConfig(
    force_remove_raw=True, include_supplementary=False,
    return_submission=True, return_tikr=True)
dataloader = edgar.DataLoader(tikrs, document_type='8-K', data_dir=DATA_DIR,
                              config=config)

# Extract presence of headers
parser = Parser_8K()

data = []
for idx, (tikr, sub, text) in enumerate(tqdm(dataloader, desc="Generating",
                                             leave=False)):

    attrs = dataloader.metadata._get_submission(tikr, sub)['attrs']
    # Remove submissions without date-time
    if 'CONFORMED PERIOD OF REPORT' not in attrs:
        continue
    date = attrs['CONFORMED PERIOD OF REPORT']

    doc_type = '8-K'
    if USE_DOC_SUBTYPES:
        # Label the type of 8-K Document
        subtypes = parser.get_section_types(text)
        subtype = None
        if len(subtypes) == 0:
            subtype = 'None'
        elif len(subtypes) == 1:
            subtype = subtypes[0]
        else:
            subtype = 'MULTI'
        doc_type = f'8-K-{subtype}'

    # Write to file
    fout.write(f"{frowidx}\t{tikr}\t{doc_type}\t{sub}\t{text}\t{date}\n")
    frowidx += 1

#  Close data pipe file
fout.close()
