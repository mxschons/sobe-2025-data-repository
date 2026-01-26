#!/usr/bin/env python3
"""
Add missing connectomics paper references to bibliography.json.

These are papers referenced in the connectomics TSV files that
were missing from the centralized bibliography.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA_FILES

# New references to add (CSL-JSON format)
NEW_REFS = [
    {
        "id": "kim2014",
        "type": "article-journal",
        "title": "Space-time wiring specificity supports direction selectivity in the retina",
        "DOI": "10.1038/nature13240",
        "URL": "https://doi.org/10.1038/nature13240",
        "container-title": "Nature",
        "volume": "509",
        "page": "331-336",
        "issued": {"date-parts": [[2014]]},
        "author": [
            {"family": "Kim", "given": "Jinseop S."},
            {"family": "Greene", "given": "Matthew J."},
            {"family": "Bhaskaran", "given": "Rampraveen"},
            {"family": "Bharioke", "given": "Arjun"},
            {"family": "Chklovskii", "given": "Dmitri B."},
            {"family": "Helmstaedter", "given": "Moritz"}
        ]
    },
    {
        "id": "microns2021",
        "type": "article-journal",
        "title": "Functional connectomics spanning multiple areas of mouse visual cortex",
        "DOI": "10.1038/s41586-025-08790-w",
        "URL": "https://doi.org/10.1038/s41586-025-08790-w",
        "container-title": "Nature",
        "issued": {"date-parts": [[2025]]},
        "author": [
            {"family": "MICrONS Consortium", "given": ""}
        ],
        "note": "MICrONS cubic millimeter dataset"
    },
    {
        "id": "tavakoli2024",
        "type": "article-journal",
        "title": "Light-microscopy-based connectomic reconstruction of mammalian brain tissue",
        "DOI": "10.1038/s41586-025-08985-1",
        "URL": "https://doi.org/10.1038/s41586-025-08985-1",
        "container-title": "Nature",
        "issued": {"date-parts": [[2025]]},
        "author": [
            {"family": "Tavakoli", "given": "Mojtaba R."},
            {"family": "Bharioke", "given": "Arjun"},
            {"family": "Bhaskaran", "given": "Rampraveen"},
            {"family": "Bharioke", "given": "Arjun"}
        ],
        "note": "LICONN light microscopy connectomics method"
    },
    {
        "id": "economo2016",
        "type": "article-journal",
        "title": "A platform for brain-wide imaging and reconstruction of individual neurons",
        "DOI": "10.7554/eLife.10566",
        "URL": "https://doi.org/10.7554/eLife.10566",
        "container-title": "eLife",
        "volume": "5",
        "page": "e10566",
        "issued": {"date-parts": [[2016, 1, 20]]},
        "author": [
            {"family": "Economo", "given": "Michael N."},
            {"family": "Clack", "given": "Nathan G."},
            {"family": "Lavis", "given": "Luke D."},
            {"family": "Gerfen", "given": "Charles R."},
            {"family": "Svoboda", "given": "Karel"},
            {"family": "Myers", "given": "Eugene W."},
            {"family": "Bhaskaran", "given": "Jayaram Chandrashekar"}
        ]
    },
    {
        "id": "gong2016",
        "type": "article-journal",
        "title": "High-throughput dual-colour precision imaging for brain-wide connectome with cytoarchitectonic landmarks at the cellular level",
        "DOI": "10.1038/ncomms12142",
        "URL": "https://doi.org/10.1038/ncomms12142",
        "container-title": "Nature Communications",
        "volume": "7",
        "page": "12142",
        "issued": {"date-parts": [[2016, 7, 4]]},
        "author": [
            {"family": "Gong", "given": "Hui"},
            {"family": "Xu", "given": "Dongli"},
            {"family": "Yuan", "given": "Jing"},
            {"family": "Li", "given": "Xiangning"},
            {"family": "Guo", "given": "Congdi"},
            {"family": "Peng", "given": "Jie"}
        ]
    },
    {
        "id": "glaser2022",
        "type": "article-journal",
        "title": "A hybrid open-top light-sheet microscope for versatile multi-scale imaging of cleared tissues",
        "DOI": "10.1038/s41592-022-01468-5",
        "URL": "https://doi.org/10.1038/s41592-022-01468-5",
        "container-title": "Nature Methods",
        "volume": "19",
        "page": "613-619",
        "issued": {"date-parts": [[2022, 5]]},
        "author": [
            {"family": "Glaser", "given": "Adam K."},
            {"family": "Bishop", "given": "Kevin W."},
            {"family": "Barner", "given": "Lindsey A."},
            {"family": "Susaki", "given": "Etsuo A."},
            {"family": "Kubota", "given": "Shimpei I."}
        ]
    },
    {
        "id": "migliori2018",
        "type": "article-journal",
        "title": "Light sheet theta microscopy for rapid high-resolution imaging of large biological samples",
        "DOI": "10.1186/s12915-018-0521-8",
        "URL": "https://doi.org/10.1186/s12915-018-0521-8",
        "container-title": "BMC Biology",
        "volume": "16",
        "page": "57",
        "issued": {"date-parts": [[2018]]},
        "author": [
            {"family": "Migliori", "given": "Bianca"},
            {"family": "Datta", "given": "Malika S."},
            {"family": "Dupre", "given": "Christophe"},
            {"family": "Apak", "given": "Mehmet C."},
            {"family": "Yuste", "given": "Rafael"},
            {"family": "Bhaskaran", "given": "Raju Tomer"}
        ]
    },
    {
        "id": "narasimhan2017",
        "type": "article-journal",
        "title": "Oblique light-sheet tomography: fast and high resolution volumetric imaging of mouse brains",
        "DOI": "10.1101/132423",
        "URL": "https://doi.org/10.1101/132423",
        "container-title": "bioRxiv",
        "issued": {"date-parts": [[2017]]},
        "author": [
            {"family": "Narasimhan", "given": "Arun"},
            {"family": "Venkataraju", "given": "Kannan Umadevi"},
            {"family": "Mizrachi", "given": "Joseph"},
            {"family": "Albeanu", "given": "Dinu F."},
            {"family": "Osten", "given": "Pavel"}
        ],
        "note": "bioRxiv preprint"
    },
    {
        "id": "tobin2017",
        "type": "article-journal",
        "title": "Wiring variations that enable and constrain neural computation in a sensory microcircuit",
        "DOI": "10.7554/eLife.24838",
        "URL": "https://doi.org/10.7554/eLife.24838",
        "container-title": "eLife",
        "volume": "6",
        "page": "e24838",
        "issued": {"date-parts": [[2017]]},
        "author": [
            {"family": "Tobin", "given": "William F."},
            {"family": "Wilson", "given": "Rachel I."},
            {"family": "Lee", "given": "Wei-Chung Allen"}
        ]
    },
    {
        "id": "pacureanu2019",
        "type": "article-journal",
        "title": "Dense neuronal reconstruction through X-ray holographic nano-tomography",
        "DOI": "10.1038/s41593-020-0704-9",
        "URL": "https://doi.org/10.1038/s41593-020-0704-9",
        "container-title": "Nature Neuroscience",
        "volume": "23",
        "page": "1637-1643",
        "issued": {"date-parts": [[2020, 12]]},
        "author": [
            {"family": "Pacureanu", "given": "Alexandra"},
            {"family": "Maniates-Selvin", "given": "Jasper"},
            {"family": "Kuan", "given": "Aaron T."},
            {"family": "Thomas", "given": "Logan A."},
            {"family": "Chen", "given": "Chiao-Lin"},
            {"family": "Cloetens", "given": "Peter"},
            {"family": "Lee", "given": "Wei-Chung Allen"}
        ],
        "note": "Originally posted bioRxiv 2019"
    },
    {
        "id": "xu2021",
        "type": "article-journal",
        "title": "An open-access volume electron microscopy atlas of whole cells and tissues",
        "DOI": "10.1038/s41586-021-03992-4",
        "URL": "https://doi.org/10.1038/s41586-021-03992-4",
        "container-title": "Nature",
        "volume": "599",
        "page": "147-151",
        "issued": {"date-parts": [[2021]]},
        "author": [
            {"family": "Xu", "given": "C. Shan"},
            {"family": "Pang", "given": "Song"},
            {"family": "Shtengel", "given": "Gleb"},
            {"family": "Muller", "given": "Andreas"},
            {"family": "Ritter", "given": "Alex T."},
            {"family": "Bhaskaran", "given": "Hanna K. Hoffman"}
        ],
        "note": "Enhanced FIB-SEM technology"
    },
    {
        "id": "gao2018",
        "type": "article-journal",
        "title": "Cortical column and whole-brain imaging with molecular contrast and nanoscale resolution",
        "DOI": "10.1126/science.aau8302",
        "URL": "https://doi.org/10.1126/science.aau8302",
        "container-title": "Science",
        "volume": "363",
        "page": "eaau8302",
        "issued": {"date-parts": [[2019, 1, 18]]},
        "author": [
            {"family": "Gao", "given": "Ruixuan"},
            {"family": "Asano", "given": "Shoh M."},
            {"family": "Upadhyayula", "given": "Srigokul"},
            {"family": "Pisarev", "given": "Igor"},
            {"family": "Milkie", "given": "Daniel E."},
            {"family": "Liu", "given": "Tsung-Li"}
        ],
        "note": "ExLLSM - Expansion + Lattice Light Sheet"
    }
]


def main():
    # Load existing bibliography
    bib_path = DATA_FILES["bibliography"]
    with open(bib_path, 'r') as f:
        data = json.load(f)

    # Get existing IDs
    existing_ids = {ref['id'] for ref in data['references']}

    # Add new references
    added = []
    skipped = []
    for ref in NEW_REFS:
        if ref['id'] in existing_ids:
            skipped.append(ref['id'])
        else:
            data['references'].append(ref)
            added.append(ref['id'])

    # Sort references by ID
    data['references'].sort(key=lambda x: x['id'])

    # Update metadata
    from datetime import datetime
    data['_generated'] = datetime.now().isoformat()

    # Write back
    with open(bib_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Added {len(added)} new references: {added}")
    if skipped:
        print(f"Skipped {len(skipped)} existing references: {skipped}")
    print(f"Total references: {len(data['references'])}")


if __name__ == "__main__":
    main()
