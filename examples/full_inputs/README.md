# Full example inputs

This directory contains the complete example input set retained from the existing local workflow.
Unlike `examples/minimal_inputs/`, these files are not reduced validation fixtures.

Use the files in the Web/API input fields as follows:

- `smiles.csv`: compound library (`ID` and `SMILES`);
- `steatosis_gene.txt`: disease-gene input;
- `TPM_matrix_1.tsv` with `metadata_1.tsv`: first expression comparison;
- `TPM_matrix_2.tsv` with `metadata_2.tsv`: second expression comparison;
- `positive_drugs.tsv`: optional KG positive-drug prior;
- `disease_links.tsv`: optional KG disease-link prior;
- `sample_metadata.tsv`: source/sample notes, not a scientific runner input.

`source.md` records the input descriptions and provenance notes carried with the original input
directory. These files are examples for reproduction and adaptation; their presence does not make
the static result in `examples/demo_result/` a newly reproduced result.

Both TPM files contain 216 sample columns. `metadata_1.tsv` selects 61 of those samples and
`metadata_2.tsv` selects 95; every selected sample is present in its paired TPM header and each
comparison contains both control and disease groups. This follows the implemented contract:
metadata samples must be a subset of the paired TPM samples, while unused TPM columns are allowed.

The complete scientific Workflow also requires the external resource bundle. Run:

```bash
python scripts/download_resources.py
python scripts/check_resources.py --mode enhanced
```

Candidate rankings remain research priorities only and require further computational and
experimental validation.
