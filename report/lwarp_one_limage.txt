@echo off
pdfseparate -f %1 -l %1 %4_html.pdf lateximages\lateximagetemp-%%d.pdf
pdfcrop  --hires  lateximages\lateximagetemp-%1.pdf lateximages\%3.pdf
pdftocairo  -svg -noshrink  lateximages\%3.pdf lateximages\%3.svg
del lateximages\%3.pdf
del lateximages\lateximagetemp-%1.pdf
exit
