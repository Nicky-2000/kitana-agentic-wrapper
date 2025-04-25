# kitana-agentic-wrapper

An agentic data search tool that leverages the output of Kitana. This wrapper adds custom agent logic for intelligent data discovery across diverse sources.

## Setup Instructions

### Step 1: Clone the Repository (with Submodules)

First run:

`git clone --recurse-submodules git@github.com:<your-username>/kitana-agentic-wrapper.git && cd kitana-agentic-wrapper`

If you've already cloned the repo, initialize submodules with:

`git submodule update --init --recursive`

### Step 2: Create a Virtual Environment and Install Dependencies

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then install dependencies:

`pip install -r kitana-e2e/requirements.txt`  
`pip install -r requirements.txt`
