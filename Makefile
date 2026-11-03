PY ?= python3

install:
	$(PY) -m pip install -r requirements.txt

demo:
	$(PY) -m aegis run --scenario all

serve:
	$(PY) -m aegis serve --port 8000

test:
	$(PY) -m pytest -q -m "not slow"

test-all:
	$(PY) -m pytest -q

modal-smoke:
	AEGIS_SANDBOX=modal $(PY) -m aegis.sandbox.modal_backend

css:
	npx tailwindcss@3.4.17 -c web/tailwind.config.js -i web/tailwind.src.css -o web/vendor/tailwind.css --minify
