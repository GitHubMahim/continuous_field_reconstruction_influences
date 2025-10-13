# Influence Visualization Tools

Three interactive tools for analyzing influences in continuous field reconstruction models:

1. **Region Tool** (`plot_region_influences.py`) - Region-to-region influences
2. **Point Tool** (`plot_point_influences.py`) - Point-to-point influences  
3. **Temporal Tool** (`plot_influence_over_time.py`) - How influences change over time

## Installation
1. Clone the project and cd into the folder.

```bash
git clone https://github.com/GitHubMahim/continuous_field_reconstruction_influences.git
cd continuous_field_reconstruction_influences
```

2. Install the required libraries to your environment.

```bash
pip install -r requirements.txt
```

3. Download and unzip the data file.

Make sure the data file is in the root directory.
| Dataset | Link |
| -------------------------------------------- | ------------------------------------------------------------ |
| All test data influences | [[Google Drive]](https://drive.google.com/file/d/1mrEu7sJ3Dc1AsR8pLDCdLtgLO973518f/view?usp=sharing) |


---

## 1. Region Influence Tool

**Files needed:** `regions_data.npy`, `yrefs.npy`

**What it does:** Shows how different spatial regions influence each other.

**4 Maps:**
- **Top Left:** Click to select a region
- **Top Right:** Shows which regions influence the selected region (purple)
- **Bottom Left:** Total influence for all regions (blue, static)
- **Bottom Right:** Shows which regions the selected region influences (grey)

**Controls:** Click regions, percentage slider (10-100%), +/- buttons

**Run:**
```bash
python plot_region_influences.py
```

---

## 2. Point Influence Tool

**Files needed:** `test_coords.npy`, `train_coords.npy`, `yrefs.npy`, `influences_all_test.npy`

**What it does:** Shows which individual training points influence a specific test location.

**2 Maps:**
- **Left:** Click to select test location (yellow circle)
- **Right:** Shows influential training points (green dots with black outlines)

**Controls:** Click locations, coordinate sliders, percentage slider (10-100%), +/- buttons

**Run:**
```bash
python plot_point_influences.py
```

---

## 3. Temporal Influence Tool

**Files needed:** `test_coords.npy`, `train_coords.npy`, `yrefs.npy`, `index_15533_influences.npy`

**What it does:** Shows how influences change over time for a fixed test point (index 15533).

**2 Maps:**
- **Left:** Reference field with fixed test point (yellow circle)
- **Right:** Influential training points at current timestamp (green dots with black outlines)

**Controls:** Timestamp slider, percentage slider (10-100%), +/- buttons

**Run:**
```bash
python plot_influence_over_time.py
```

---

## Troubleshooting

- **FileNotFoundError:** Check all required .npy files are present
- **Empty plots:** Verify influence data contains valid values
- **ImportError:** Run `pip install -r requirements.txt`
- **Coordinates:** Must be normalized between 0 and 1

## Quick Start

1. Install requirements: `pip install -r requirements.txt`
2. Ensure data files are in the same directory
3. Run any tool: `python plot_[tool_name].py`
4. Use interactive controls to explore influences
