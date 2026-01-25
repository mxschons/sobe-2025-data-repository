# Brain Emulation Report 2025 - Makefile
# Common tasks for figure generation and build pipeline

.PHONY: help install figures downloads validate serve clean all calculator-install calculator-build calculator-test calculator-clean

PYTHON ?= python3
SCRIPTS_DIR = scripts
OUTPUT_DIR = data-and-figures

help:
	@echo "Brain Emulation Report 2025 - Build Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install     Install Python dependencies"
	@echo "  figures     Generate all figures"
	@echo "  downloads   Build ZIP archives for download"
	@echo "  validate    Run quality checks"
	@echo "  serve       Start local development server"
	@echo "  clean       Remove generated files"
	@echo "  all         Run full pipeline (figures + calculator + downloads)"
	@echo ""
	@echo "Calculator targets:"
	@echo "  calculator-install  Install calculator dependencies"
	@echo "  calculator-build    Build calculator outputs"
	@echo "  calculator-test     Run calculator tests"
	@echo "  calculator-clean    Remove calculator outputs"
	@echo ""

install:
	$(PYTHON) -m pip install -r requirements.txt

figures:
	cd $(SCRIPTS_DIR) && $(PYTHON) run_all_figures.py

downloads:
	cd $(SCRIPTS_DIR) && $(PYTHON) build_downloads.py

validate:
	cd $(SCRIPTS_DIR) && $(PYTHON) validate.py

serve:
	@echo "Starting server at http://localhost:8000"
	cd $(OUTPUT_DIR) && $(PYTHON) -m http.server 8000

clean:
	rm -rf $(OUTPUT_DIR)/figures/generated/*.png
	rm -rf $(OUTPUT_DIR)/figures/generated/*.svg
	rm -rf $(OUTPUT_DIR)/figures/generated/*.webp
	rm -rf $(OUTPUT_DIR)/figures/generated/*.avif
	rm -rf $(OUTPUT_DIR)/figures/generated/neuro-sim/*
	rm -rf $(OUTPUT_DIR)/figures/generated/neuro-rec/*
	rm -rf $(OUTPUT_DIR)/figures/generated/radar-charts/*
	rm -rf $(OUTPUT_DIR)/downloads/*.zip

all: figures calculator-build downloads
	@echo ""
	@echo "Build complete!"
	@echo "Output in: $(OUTPUT_DIR)/"

# Calculator targets
calculator-install:
	cd $(SCRIPTS_DIR)/calculator && npm ci

calculator-build:
	cd $(SCRIPTS_DIR)/calculator && npm run build:data

calculator-test:
	cd $(SCRIPTS_DIR)/calculator && npm test

calculator-clean:
	rm -rf $(OUTPUT_DIR)/calculator/
