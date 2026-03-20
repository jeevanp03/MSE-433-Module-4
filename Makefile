.PHONY: all analysis dashboard serve clean

all: analysis dashboard

analysis:
	source venv/bin/activate && python3 main.py

dashboard: analysis
	source venv/bin/activate && python3 app/backend/export_dashboard_data.py
	source venv/bin/activate && python3 app/backend/whatif_simulator.py
	source venv/bin/activate && python3 app/backend/reassignment_data.py

serve:
	cd app/frontend && npm run dev

clean:
	rm -rf output/ app/frontend/src/data/dashboard_data.json app/frontend/src/data/whatif_data.json app/frontend/src/data/reassignment_data.json app/frontend/dist/
