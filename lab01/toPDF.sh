#!/bin/bash

rm -rf output.pdf
pandoc *.md -o output.pdf --pdf-engine=xelatex
