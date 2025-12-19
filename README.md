# Text Tree Edit Distance: A Language Model-Based Metric for Text Hierarchies.

## Abstract
Text trees as a data structure occur in numerous machine learning tasks like hierarchical summarization and automatic mind map generation. One of the main methods of quality evaluation in these tasks is comparison with reference hierarchies created by experts. The method used so far to compare text hierarchies, as shown in this work, poorly accounts for their structure and text semantics relative to phrasing. To address this issue, we propose a new metric on the set of text trees — text tree edit distance (TTED), based on tree edit distance with semantic distance between texts measured using a large language model. To evaluate how the metric reflects different aspects of text tree difference, we introduce special quality coefficients that reflect the sensitivity of a metric to paraphrasing relative to structural and semantic differences of text trees. Using these coefficients, we conduct extensive testing of the proposed metric and its modifications compared to a baseline used in previous works to compare text hierarchies, which shows that TTED indeed captures significant differences between text trees more accurately than the previously used method. We also provide a practical implementation of TTED for further usage.

## Installation & Usage
All the code for this project can be found in the [`code`](https://github.com/intsystems/Sobolevsky-MS-Thesis/code) directory of this repository.

It is recommended to use a fresh virtual environment of choice. For example:
```
python -m venv tted
source tted/bin/activate # for Linux
tted/bin/activate.bat # for Windows
```

The code and required dependencies can be install with the following code:
```
git clone https://github.com/intsystems/Sobolevsky-MS-Thesis
cd ./Sobolevsky-MS-Thesis/code
pip install -r requirements.txt
```

All the experiments for TTED can be found in [`tted_tests.ipynb`](https://github.com/intsystems/Sobolevsky-MS-Thesis/code/tted_tests.ipynb). The source code for TTED can be found in the [`/tted`](https://github.com/intsystems/Sobolevsky-MS-Thesis/code/tted) subdirectory. The data and prompts used for the experiments are located in the [`/data`](https://github.com/intsystems/Sobolevsky-MS-Thesis/code/data) subdirectory.

## Publication
- F. Sobolevsky and K. Vorontsov, «Text Tree Edit Distance: A Language Model-Based Metric for Text Hierarchies», _2025 IEEE XVII International Scientific and Technical Conference on Actual Problems of Electronic Instrument Engineering (APEIE), Novosibirsk, Russian Federation, 2025, pp. 1-5, doi: 10.1109/APEIE66761.2025.11289395._

## Citation
BibTeX:
```bibtex
@INPROCEEDINGS{11289395,
  author={Sobolevsky, Fedor and Vorontsov, Konstantin},
  booktitle={2025 IEEE XVII International Scientific and Technical Conference on Actual Problems of Electronic Instrument Engineering (APEIE)}, 
  title={Text Tree Edit Distance: A Language Model-Based Metric for Text Hierarchies}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Sensitivity;Trees (botanical);Large language models;Instruments;Semantics;Machine learning;Data structures;Testing;text trees;mind map;hierarchical summarization;tree edit distance;large language models;Zhang-Shasha algorithm},
  doi={10.1109/APEIE66761.2025.11289395}}
```
