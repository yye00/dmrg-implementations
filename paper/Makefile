MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean figures

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

figures: figures/fig6_ablation.pdf

figures/fig6_ablation.pdf: ../benchmarks/scripts/generate_ablation_figure.py
	cd ../benchmarks/scripts && python generate_ablation_figure.py --out ../../paper/figures/fig6_ablation.pdf

clean:
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).bbl $(MAIN).blg \
	      $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).spl \
	      $(MAIN).fls $(MAIN).fdb_latexmk
