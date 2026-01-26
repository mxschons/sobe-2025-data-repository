#!/usr/bin/env python3
"""
Add remaining connectomics paper references to bibliography.json.

These are papers referenced in the connectomics TSV files that
were missing from the centralized bibliography after the first batch.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA_FILES

# New references to add (CSL-JSON format)
NEW_REFS = [
    {
        "id": "voleti2019",
        "type": "article-journal",
        "title": "Real-time volumetric microscopy of in vivo dynamics and large-scale samples with SCAPE 2.0",
        "DOI": "10.1038/s41592-019-0579-4",
        "URL": "https://doi.org/10.1038/s41592-019-0579-4",
        "container-title": "Nature Methods",
        "volume": "16",
        "page": "1054-1062",
        "issued": {"date-parts": [[2019, 10]]},
        "author": [
            {"family": "Voleti", "given": "Venkatakaushik"},
            {"family": "Patel", "given": "Kripa B."},
            {"family": "Li", "given": "Wenze"},
            {"family": "Perez Campos", "given": "Citlali"},
            {"family": "Bharadwaj", "given": "Srinidhi"},
            {"family": "Hillman", "given": "Elizabeth M. C."}
        ],
        "note": "SCAPE 2.0 swept confocally-aligned planar excitation microscopy"
    },
    {
        "id": "wang2021",
        "type": "article-journal",
        "title": "Chemical sectioning fluorescence tomography: high-throughput, high-contrast, multicolor, whole-brain imaging at subcellular resolution",
        "DOI": "10.1016/j.celrep.2021.108709",
        "URL": "https://doi.org/10.1016/j.celrep.2021.108709",
        "container-title": "Cell Reports",
        "volume": "34",
        "page": "108709",
        "issued": {"date-parts": [[2021, 2]]},
        "author": [
            {"family": "Wang", "given": "Xiaojun"},
            {"family": "Tomer", "given": "Raju"}
        ],
        "note": "fMOST/CSFT chemical sectioning fluorescence tomography"
    },
    {
        "id": "glaser2023",
        "type": "article-journal",
        "title": "Expansion-assisted selective plane illumination microscopy for nanoscale imaging of centimeter-scale tissues",
        "DOI": "10.7554/eLife.91979",
        "URL": "https://doi.org/10.7554/eLife.91979",
        "container-title": "eLife",
        "volume": "12",
        "page": "RP91979",
        "issued": {"date-parts": [[2025]]},
        "author": [
            {"family": "Glaser", "given": "Adam K."},
            {"family": "Chandrashekar", "given": "Jayaram"},
            {"family": "Vasquez", "given": "Joshua"},
            {"family": "Arshadi", "given": "Cameron"},
            {"family": "Ouellette", "given": "Naveen"}
        ],
        "note": "ExA-SPIM expansion-assisted selective plane illumination microscopy"
    },
    {
        "id": "vladimirov2024",
        "type": "article-journal",
        "title": "Benchtop mesoSPIM: a next-generation open-source light-sheet microscope for cleared samples",
        "DOI": "10.1038/s41467-024-46770-2",
        "URL": "https://doi.org/10.1038/s41467-024-46770-2",
        "container-title": "Nature Communications",
        "volume": "15",
        "page": "2679",
        "issued": {"date-parts": [[2024, 3, 27]]},
        "author": [
            {"family": "Vladimirov", "given": "Nikita"},
            {"family": "Voigt", "given": "Fabian F."},
            {"family": "Naert", "given": "Thomas"},
            {"family": "Araujo", "given": "Gabriela R."},
            {"family": "Cai", "given": "Ruiyao"},
            {"family": "Helmchen", "given": "Fritjof"}
        ],
        "note": "Benchtop mesoSPIM open-source light-sheet microscope"
    },
    {
        "id": "sievers2024",
        "type": "article-journal",
        "title": "RoboEM: automated 3D flight tracing for synaptic-resolution connectomics",
        "DOI": "10.1038/s41592-024-02226-5",
        "URL": "https://doi.org/10.1038/s41592-024-02226-5",
        "container-title": "Nature Methods",
        "volume": "21",
        "page": "908-913",
        "issued": {"date-parts": [[2024, 5]]},
        "author": [
            {"family": "Schmidt", "given": "Martin"},
            {"family": "Motta", "given": "Alessandro"},
            {"family": "Sievers", "given": "Meike"},
            {"family": "Helmstaedter", "given": "Moritz"}
        ],
        "note": "RoboEM AI-based flight tracing for mSEM connectomics"
    },
    {
        "id": "wildenberg2024",
        "type": "article-journal",
        "title": "Laminography as a tool for imaging large-size samples with high resolution",
        "DOI": "10.1107/S1600577524002923",
        "URL": "https://doi.org/10.1107/S1600577524002923",
        "container-title": "Journal of Synchrotron Radiation",
        "volume": "31",
        "page": "851-866",
        "issued": {"date-parts": [[2024, 7, 1]]},
        "author": [
            {"family": "Nikitin", "given": "Viktor"},
            {"family": "Wildenberg", "given": "Gregg"},
            {"family": "Mittone", "given": "Alberto"},
            {"family": "Shevchenko", "given": "Pavel"},
            {"family": "De Carlo", "given": "Francesco"}
        ],
        "note": "Laminography pipeline for µXCT imaging of large brain samples"
    },
    {
        "id": "hoffman2020",
        "type": "article-journal",
        "title": "Correlative three-dimensional super-resolution and block-face electron microscopy of whole vitreously frozen cells",
        "DOI": "10.1126/science.aaz5357",
        "URL": "https://doi.org/10.1126/science.aaz5357",
        "container-title": "Science",
        "volume": "367",
        "page": "eaaz5357",
        "issued": {"date-parts": [[2020, 1, 17]]},
        "author": [
            {"family": "Hoffman", "given": "David P."},
            {"family": "Shtengel", "given": "Gleb"},
            {"family": "Xu", "given": "C. Shan"},
            {"family": "Campbell", "given": "Kirsten R."},
            {"family": "Freeman", "given": "Melanie"},
            {"family": "Hess", "given": "Harald F."}
        ],
        "note": "Correlative super-resolution and FIB-SEM imaging"
    },
    {
        "id": "yin2020",
        "type": "article-journal",
        "title": "A petascale automated imaging pipeline for mapping neuronal circuits with high-throughput transmission electron microscopy",
        "DOI": "10.1038/s41467-020-18659-3",
        "URL": "https://doi.org/10.1038/s41467-020-18659-3",
        "container-title": "Nature Communications",
        "volume": "11",
        "page": "4949",
        "issued": {"date-parts": [[2020, 10, 2]]},
        "author": [
            {"family": "Yin", "given": "Wenjing"},
            {"family": "Brittain", "given": "Derrick"},
            {"family": "Borseth", "given": "Jay"},
            {"family": "Scott", "given": "Marie E."},
            {"family": "Williams", "given": "Derric"},
            {"family": "da Costa", "given": "Nuno Macarico"}
        ],
        "note": "autoTEM/piTEAM petascale automated imaging pipeline"
    },
    {
        "id": "zheng2022",
        "type": "article-journal",
        "title": "Fast imaging of millimeter-scale areas with beam deflection transmission electron microscopy",
        "DOI": "10.1038/s41467-024-50846-4",
        "URL": "https://doi.org/10.1038/s41467-024-50846-4",
        "container-title": "Nature Communications",
        "volume": "15",
        "page": "6860",
        "issued": {"date-parts": [[2024]]},
        "author": [
            {"family": "Zheng", "given": "Zhihao"},
            {"family": "Own", "given": "Christopher S."},
            {"family": "Wanner", "given": "Adrian A."},
            {"family": "Koene", "given": "Randal A."},
            {"family": "Seung", "given": "H. Sebastian"}
        ],
        "note": "bdTEM beam deflection transmission electron microscopy"
    },
    {
        "id": "allenlee2016",
        "type": "article-journal",
        "title": "Anatomy and function of an excitatory network in the visual cortex",
        "DOI": "10.1038/nature17192",
        "URL": "https://doi.org/10.1038/nature17192",
        "container-title": "Nature",
        "volume": "532",
        "page": "370-374",
        "issued": {"date-parts": [[2016, 4, 21]]},
        "author": [
            {"family": "Lee", "given": "Wei-Chung Allen"},
            {"family": "Bonin", "given": "Vincent"},
            {"family": "Reed", "given": "Michael"},
            {"family": "Graham", "given": "Brett J."},
            {"family": "Hood", "given": "Greg"},
            {"family": "Reid", "given": "R. Clay"}
        ],
        "note": "Visual cortex network anatomy with ss-TEMCA"
    },
    {
        "id": "humbel2024",
        "type": "article-journal",
        "title": "Synchrotron radiation-based tomography of an entire mouse brain with sub-micron voxels: augmenting interactive brain atlases with terabyte data",
        "DOI": "10.1002/advs.202416879",
        "URL": "https://doi.org/10.1002/advs.202416879",
        "container-title": "Advanced Science",
        "issued": {"date-parts": [[2025]]},
        "author": [
            {"family": "Humbel", "given": "Mattia"},
            {"family": "Tanner", "given": "Christine"},
            {"family": "Girona Alarcón", "given": "Marta"},
            {"family": "Schulz", "given": "Georg"},
            {"family": "Müller", "given": "Bert"},
            {"family": "Rodgers", "given": "Griffin"}
        ],
        "note": "Whole mouse brain µXCT at 0.65 µm voxel resolution"
    },
    {
        "id": "chakraborty2019",
        "type": "article-journal",
        "title": "Light-sheet microscopy of cleared tissues with isotropic, subcellular resolution",
        "DOI": "10.1038/s41592-019-0615-4",
        "URL": "https://doi.org/10.1038/s41592-019-0615-4",
        "container-title": "Nature Methods",
        "volume": "16",
        "page": "1109-1113",
        "issued": {"date-parts": [[2019, 11]]},
        "author": [
            {"family": "Chakraborty", "given": "Tonmoy"},
            {"family": "Driscoll", "given": "Meghan K."},
            {"family": "Jeffery", "given": "Elise"},
            {"family": "Murphy", "given": "Malea M."},
            {"family": "Roudot", "given": "Philippe"},
            {"family": "Bhaskaran", "given": "Bo-Jui Chang"}
        ],
        "note": "ctASLM cleared-tissue axially swept light-sheet microscopy"
    },
    {
        "id": "zhang2021",
        "type": "article-journal",
        "title": "Multi-Scale Light-Sheet Fluorescence Microscopy for Fast Whole Brain Imaging",
        "DOI": "10.3389/fnana.2021.732464",
        "URL": "https://doi.org/10.3389/fnana.2021.732464",
        "container-title": "Frontiers in Neuroanatomy",
        "volume": "15",
        "page": "732464",
        "issued": {"date-parts": [[2021, 9, 24]]},
        "author": [
            {"family": "Zhang", "given": "Zhouzhou"},
            {"family": "Yao", "given": "Xiao"},
            {"family": "Yin", "given": "Xinxin"},
            {"family": "Ding", "given": "Zhangcan"},
            {"family": "Huang", "given": "Tianyi"},
            {"family": "Guo", "given": "Zengcai V."}
        ],
        "note": "mLSFM multi-scale light-sheet fluorescence microscopy"
    },
    {
        "id": "prince2024",
        "type": "article-journal",
        "title": "Signal improved ultra-fast light-sheet microscope for large tissue imaging",
        "DOI": "10.1038/s44172-024-00205-4",
        "URL": "https://doi.org/10.1038/s44172-024-00205-4",
        "container-title": "Communications Engineering",
        "volume": "3",
        "page": "52",
        "issued": {"date-parts": [[2024, 4, 2]]},
        "author": [
            {"family": "Prince", "given": "Md Nasful Huda"},
            {"family": "Bhaskaran", "given": "Rampraveen"}
        ],
        "note": "Ultra-fast axially swept light-sheet microscopy (ALSM)"
    },
    {
        "id": "tang2024",
        "type": "article-journal",
        "title": "Curved light sheet microscopy for centimeter-scale cleared tissues imaging",
        "DOI": "10.1101/2024.04.28.591483",
        "URL": "https://doi.org/10.1101/2024.04.28.591483",
        "container-title": "bioRxiv",
        "issued": {"date-parts": [[2024, 4, 29]]},
        "author": [
            {"family": "Tang", "given": "Lijuan"},
            {"family": "Wang", "given": "Jiayu"},
            {"family": "Ding", "given": "Jiayi"},
            {"family": "Sun", "given": "Junyou"},
            {"family": "Wu", "given": "Jianglai"}
        ],
        "note": "Curved LSFM for centimeter-scale tissue imaging"
    },
    {
        "id": "qi2023",
        "type": "article-journal",
        "title": "Confocal Airy beam oblique light-sheet tomography for brain-wide cell type distribution and morphology",
        "DOI": "10.1038/s41592-025-02888-9",
        "URL": "https://doi.org/10.1038/s41592-025-02888-9",
        "container-title": "Nature Methods",
        "issued": {"date-parts": [[2025]]},
        "author": [
            {"family": "Qi", "given": "Xiaoli"},
            {"family": "Muñoz-Castañeda", "given": "Rodrigo"},
            {"family": "Narasimhan", "given": "Arun"},
            {"family": "Ding", "given": "Liya"},
            {"family": "Osten", "given": "Pavel"}
        ],
        "note": "CAB-OLST confocal Airy beam oblique light-sheet tomography"
    },
    {
        "id": "wang2024",
        "type": "article-journal",
        "title": "Mesoscale volumetric fluorescence imaging at nanoscale resolution by photochemical sectioning",
        "DOI": "10.1126/science.adr9109",
        "URL": "https://doi.org/10.1126/science.adr9109",
        "container-title": "Science",
        "volume": "390",
        "page": "eadr9109",
        "issued": {"date-parts": [[2025]]},
        "author": [
            {"family": "Wang", "given": "Wei"},
            {"family": "Ruan", "given": "Xiongtao"},
            {"family": "Liu", "given": "Gaoxiang"},
            {"family": "Milkie", "given": "Daniel E."},
            {"family": "Betzig", "given": "Eric"},
            {"family": "Gao", "given": "Ruixuan"}
        ],
        "note": "VIPS volumetric imaging via photochemical sectioning"
    },
    {
        "id": "guo2020",
        "type": "article-journal",
        "title": "Rapid image deconvolution and multiview fusion for optical microscopy",
        "DOI": "10.1038/s41587-020-0560-x",
        "URL": "https://doi.org/10.1038/s41587-020-0560-x",
        "container-title": "Nature Biotechnology",
        "volume": "38",
        "page": "1337-1346",
        "issued": {"date-parts": [[2020]]},
        "author": [
            {"family": "Guo", "given": "Min"},
            {"family": "Li", "given": "Yue"},
            {"family": "Su", "given": "Yicong"},
            {"family": "Lambert", "given": "Talley"},
            {"family": "Bhaskaran", "given": "Damian Dalle Nogare"},
            {"family": "Bhaskaran", "given": "Harshad Nagaraj"}
        ],
        "note": "Dual-inverted SPIM deconvolution and multiview fusion"
    },
    {
        "id": "chen2023",
        "type": "article-journal",
        "title": "Low-cost and scalable projected light-sheet microscopy for the high-resolution imaging of cleared tissue and living samples",
        "DOI": "10.1038/s41551-024-01249-9",
        "URL": "https://doi.org/10.1038/s41551-024-01249-9",
        "container-title": "Nature Biomedical Engineering",
        "volume": "8",
        "page": "1109-1123",
        "issued": {"date-parts": [[2024, 9]]},
        "author": [
            {"family": "Chen", "given": "Yannan"},
            {"family": "Gong", "given": "Cheng"},
            {"family": "Chauhan", "given": "Shradha"},
            {"family": "De La Cruz", "given": "Estanislao Daniel"},
            {"family": "Datta", "given": "Malika S."},
            {"family": "Tomer", "given": "Raju"}
        ],
        "note": "pLSM projected light-sheet microscopy"
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
