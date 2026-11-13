# SnapCal manuscript (IEEE Embedded Systems Letters)

Target: [IEEE Embedded Systems Letters](https://ieee-ceda.org/publication/esl/esl-author-instructions)
(IEEE transactions two-column, ≤4 pages including references; `IEEEtran` journal).
Prior Sensors Letters freeze: `LSENS_20261108/` (desk-rejected out of scope,
SENSL-26-11-RL-1722). ESL pack: `ESL_20261111/`.
No arXiv preprint. Code and tabulated results:
https://github.com/T01284/SnapCal

## Layout

```
paper/
├── template/     IEEE_lsens.cls, IEEEtran.bst, logo
├── draft/        snapcal.tex, refs.bib, build.ps1
├── figures/      PDF figures from gen_figures.py
├── data/         JSON tables used by the letter
└── scripts/      gen_figures.py, pack_github_repo.py
```

## Build

```powershell
cd paper\draft
.\build.ps1
```

Regenerate figures from `results/`:

```powershell
python paper/scripts/gen_figures.py
```

Flat submission pack: repository root `LSENS_YYYYMMDD/`
(tracked freeze + `.gitignore`; see that folder's `README.txt`).

```powershell
python paper/scripts/pack_github_repo.py      # -> LSENS_*/github_repo.zip (tracked)
python paper/scripts/pack_latex_source.py     # -> LSENS_*/latex_source.zip (tracked)
python paper/scripts/gen_graphical_abstract.py
```

## Authors

Li Zhao (Heilongjiang Jiaotong Polytechnic), Ming Zhao (Harbin Cambridge
University), Shu Wang (Heilongjiang Tianyouwei Electronics Co., Ltd.;
corresponding author, shu.wang@hljtyw.com).
