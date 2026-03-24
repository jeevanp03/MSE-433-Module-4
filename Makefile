.PHONY: all analysis dashboard serve clean

# Use uv if available, otherwise fall back to python3
PYTHON := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "python3")

all: analysis dashboard

analysis:
	$(PYTHON) main.py

dashboard: analysis
	$(PYTHON) app/backend/export_dashboard_data.py
	$(PYTHON) app/backend/whatif_simulator.py
	$(PYTHON) app/backend/reassignment_data.py

serve:
	cd app/frontend && npm run dev

clean:
	rm -rf output/ app/frontend/src/data/dashboard_data.json app/frontend/src/data/whatif_data.json app/frontend/src/data/reassignment_data.json app/frontend/dist/
