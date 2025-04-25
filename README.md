# kitana-agentic-wrapper

An agentic data search tool that leverages the output of Kitana. This wrapper adds custom agent logic for intelligent data discovery across diverse sources.

## Setup Instructions

### Step 1: Clone the Repository (with Submodules)

First run:
```bash
git clone --recurse-submodules git@github.com:<your-username>/kitana-agentic-wrapper.git && cd kitana-agentic-wrapper
```

If you've already cloned the repo, initialize submodules with:

```bash
git submodule update --init --recursive
```

### Step 2: Create a Virtual Environment and Install Dependencies

Create and activate a virtual environment:

NOTE: Use Python 3.9! Python 3.12.6 is known not to work!

```bash
python3 -m venv venv
source venv/bin/activate
```

Then install dependencies:

```bash
pip install -r kitana-e2e/requirements.txt
`pip install -r requirements.txt
```

NOTE: To make IntelliSense work for the submodule you can manually add the submodule path in VSCode

1. Open `.vscode/settings.json`
2. Add the following:

```json
{
  "python.envFile": "${workspaceFolder}/.env",
  "python.analysis.extraPaths": ["kitana-e2e"]
}
```

### Running the code: 

```bash
python3 run_kitana_agentic_wrapper.py

```

