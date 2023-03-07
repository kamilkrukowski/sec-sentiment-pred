import edgar
from .parser_8k import Parser_8K

DATA_DIR = 'data'
# Whether to include type of 8-K document in embedding.
USE_DOC_SUBTYPES = True

tikrs = ['aapl', 'msft', 'amzn', 'tsla', 'googl', 'goog',  'unh', 'jnj', 'cvx',
         'jpm', 'hd', 'v', 'pg']

#  Prepare file for constant offloading
fout = open('8k_data.csv', 'w')
fout.write(",tikr,FORM_TYPE,submission,text,Date\n")
frowidx = 0

config = edgar.DataLoaderConfig(
    force_remove_raw=True, include_supplementary=False,
    return_submission=True, return_tikr=True)
dataloader = edgar.DataLoader(tikrs, document_type='8-K', data_dir=DATA_DIR,
                              config=config)

# Extract presence of headers
parser = Parser_8K()

data = []
for idx, tikr, sub, text in enumerate(dataloader):

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
    fout.write(f"{frowidx},{tikr},{doc_type},{sub},{text},{date}\n")
    frowidx += 1

#  Close data pipe file
fout.close()
