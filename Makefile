PY ?= python3
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

SMOKE := --model configs/model/debug_tiny.yaml --exp configs/exp/smoke.yaml --no-progress
STAGES := 01_baselines 02_mean_activations 03_mediation 04_build_fv \
          05_intervene 06_transfer 07_logitlens 08_controls

.PHONY: help lint smoke sentences paper slides clean clean-results

help:
	@echo "make lint          check the hand-edited TSVs"
	@echo "make smoke         whole pipeline on a tiny CPU model (no GPU, no weights)"
	@echo "make sentences     regenerate data/part2/sentences.tsv from the source tables"
	@echo "make paper         build paper/main.pdf (LuaLaTeX)"
	@echo "make slides        build presentation/main.pdf (LuaLaTeX)"
	@echo "make clean         remove LaTeX build artefacts"
	@echo "make clean-results remove cached results (forces a full recompute)"

lint:
	$(PY) scripts/lint_data.py

smoke:
	$(PY) scripts/00_env_check.py $(SMOKE) --skip-gpu
	@for s in $(STAGES); do \
	  printf '%-24s ' $$s; \
	  $(PY) scripts/$$s.py $(SMOKE) >/dev/null || { echo FAIL; exit 1; }; \
	  echo OK; \
	done
	@$(PY) scripts/03_mediation.py $(SMOKE) --pooled >/dev/null && echo "pooled mediation      OK"
	@$(PY) scripts/09_report.py $(SMOKE) >/dev/null && echo "09_report             OK"

sentences:
	$(PY) -c "from fvcm.data import write_sentences; print(write_sentences())"

# LuaLaTeX, not pdfLaTeX: pdfLaTeX renders every Telugu glyph blank.
paper:
	cd paper && latexmk -lualatex -interaction=nonstopmode main.tex

slides:
	cd presentation && latexmk -lualatex -interaction=nonstopmode main.tex

clean:
	cd paper && latexmk -C || true
	cd presentation && latexmk -C || true

clean-results:
	rm -rf results/cache results/aie results/fv
	@echo "cached results removed; the next run recomputes from scratch"
