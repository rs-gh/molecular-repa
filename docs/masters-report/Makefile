# requires GNU make
SHELL=/bin/bash

.DELETE_ON_ERROR:

%.pdf %.aux %.idx: %.tex
	pdflatex -halt-on-error -file-line-error $<
	while grep 'Rerun to get ' $*.log ; do pdflatex -halt-on-error $< ; done
%.ind: %.idx
	makeindex $*
%.bbl: %.aux
	bibtex $*
%.pdftex %.pdftex_t: %.fig
	fig2dev -L pdftex_t -p $*.pdftex $< $*.pdftex_t
	fig2dev -L pdftex $< $*.pdftex

all: report.pdf report-submission.pdf

.PHONY: pdf
pdf: report-draft.pdf

report-submission.tex: report.tex
	sed -e 's/^%\(\\submissiontrue\)/\1/' $< >$@

# included assets: any change to a table/figure .tex source, or to a figure
# image asset, must force a rebuild of the draft. Without these prerequisites
# make sees report-draft.tex unchanged and treats the PDF as up to date.
DRAFT_DEPS := $(wildcard tables/*.tex) \
              $(wildcard figures/*.tex) \
              $(wildcard figures/*.png) $(wildcard figures/*.pdf) \
              $(wildcard figures/*.jpg) $(wildcard figures/*.jpeg)

report.pdf: logo-dcst-colour.pdf
report-draft.pdf: logo-dcst-colour.pdf $(DRAFT_DEPS)

# ============================================================================
# Submission deliverables (both built from the single source report-draft.tex)
#
#   make submissions    -> builds BOTH PDFs below from the current draft
#   make report-final.pdf   NORMAL copy: title page with name/college/date,
#                           signed declaration, acknowledgements page.
#   make report-anon.pdf    ANONYMISED copy for blind marking: candidate-number
#                           cover sheet, no name, no signature, no
#                           acknowledgements.
#
# Both versions share one source and are byte-identical in the body. They
# differ only in the front matter, switched by the \ifsubmission boolean in
# report-draft.tex. The anonymised build is made from a throwaway copy
# (report-anon.tex) with \submissiontrue uncommented; report-anon.tex is a
# generated, disposable artefact.
# ============================================================================
.PHONY: submissions
submissions: report-final.pdf report-anon.pdf

# NORMAL copy == the draft built with the submission flag off (its default).
report-final.pdf: report-draft.pdf
	cp -f report-draft.pdf $@

# Generated source for the anonymised build: uncomment \submissiontrue.
report-anon.tex: report-draft.tex
	sed -e 's/^%\(\\submissiontrue\)/\1/' $< >$@

# Explicit recipe (overrides the generic %.pdf rule) so the multi-pass build
# resolves the cover-sheet page-count cross-references in the anon layout.
report-anon.pdf: report-anon.tex logo-dcst-colour.pdf $(DRAFT_DEPS)
	pdflatex -halt-on-error -file-line-error report-anon.tex
	while grep -q 'Rerun to get ' report-anon.log ; do pdflatex -halt-on-error report-anon.tex ; done

WORDCOUNT_FILE=report-draft

# ---- word count (texcount over the LaTeX source) ----
# texcount is vendored at tools/texcount.pl for reproducibility (no system
# install needed). Scope = core chapters only: the %TC:ignore / %TC:endignore
# directives in $(WORDCOUNT_FILE).tex exclude the front matter, bibliography,
# and appendix. -inc follows \input{} table/figure files so their captions are
# counted. The reported figure is the Sum count: body text + section headings
# + figure/table captions (narrative text). texcount does not count tabular
# cell data, matching the ACS "excluding ... data listings" clause.
TEXCOUNT=perl tools/texcount.pl

.PHONY: wordcount wordcount-detail wordcount-gs
wordcount:
	@$(TEXCOUNT) -inc -sum -total -1 $(WORDCOUNT_FILE).tex

# Full breakdown (text / headers / captions, per file).
wordcount-detail:
	@$(TEXCOUNT) -inc -sum $(WORDCOUNT_FILE).tex

# Audit only: ghostscript page-extraction over the core page range. NOTE the
# `grep -a` -- without it, ligature bytes (fi, fl) make grep treat the stream as
# binary and silently undercount. This counts rendered page furniture (running
# headers, page numbers, table cell data), so it OVER-counts vs texcount; kept
# purely for cross-checking. Requires ghostscript.
FIRSTPAGE?=$(shell sed -ne 's/^\\newlabel{firstcontentpage}{{[0-9\.]*}{\([0-9]*\)}.*/\1/p' $(WORDCOUNT_FILE).aux)
LASTPAGE ?=$(shell sed -ne 's/^\\newlabel{lastcontentpage}{{[0-9\.]*}{\([0-9]*\)}.*/\1/p' $(WORDCOUNT_FILE).aux)
wordcount-gs: $(WORDCOUNT_FILE).pdf
	@gs -q -dSAFER -sDEVICE=txtwrite -o - \
	   -dFirstPage=$(FIRSTPAGE) -dLastPage=$(LASTPAGE) $< | \
	LC_ALL=C grep -a -E '[A-Za-z]{3}' | wc -w

clean:
	rm -f *.log *.aux *.toc *.bbl *.ind *.lot *.lof *.out *~
	rm -f report-submission.tex report-anon.tex
