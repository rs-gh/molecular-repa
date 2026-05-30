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

# extract number of first and last page of the main chapters from the AUX file
WORDCOUNT_FILE=report-draft
FIRSTPAGE?=$(shell sed -ne 's/^\\newlabel{firstcontentpage}{{[0-9\.]*}{\([0-9]*\)}.*/\1/p' $(WORDCOUNT_FILE).aux)
LASTPAGE ?=$(shell sed -ne 's/^\\newlabel{lastcontentpage}{{[0-9\.]*}{\([0-9]*\)}.*/\1/p' $(WORDCOUNT_FILE).aux)

# requires ghostscript
wordcount: $(WORDCOUNT_FILE).pdf
	gs -q -dSAFER -sDEVICE=txtwrite -o - \
	   -dFirstPage=$(FIRSTPAGE) -dLastPage=$(LASTPAGE) $< | \
	egrep '[A-Za-z]{3}' | wc -w

clean:
	rm -f *.log *.aux *.toc *.bbl *.ind *.lot *.lof *.out *~
	rm -f report-submission.tex
