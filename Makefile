.PHONY: all analysis dashboard serve clean

all: analysis dashboard

analysis:
	uv run main.py

dashboard: analysis
	uv run app/backend/export_dashboard_data.py
	uv run app/backend/whatif_simulator.py
	uv run app/backend/reassignment_data.py

serve:
	cd app/frontend && npm run dev

clean:
	rm -rf output/ app/frontend/src/data/dashboard_data.json app/frontend/src/data/whatif_data.json app/frontend/src/data/reassignment_data.json app/frontend/dist/
