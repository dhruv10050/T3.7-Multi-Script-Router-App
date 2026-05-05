"""Generate report.pdf from report HTML using WeasyPrint."""
import sys
from pathlib import Path
from weasyprint import HTML, CSS

SRC = Path(__file__).parent
HTML_FILE = SRC / "report.html"
OUT = SRC / "report.pdf"

html = HTML(filename=str(HTML_FILE), base_url=str(SRC))
css  = CSS(string="""
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
body { font-family: 'Times New Roman', Times, serif; font-size: 10pt;
       line-height: 1.45; color: #111; }
h1 { font-size: 16pt; text-align: center; margin-bottom: 4pt; }
h2 { font-size: 12pt; margin-top: 12pt; margin-bottom: 4pt; border-bottom: 1px solid #ccc; }
h3 { font-size: 10.5pt; margin-top: 8pt; margin-bottom: 2pt; }
h4 { font-size: 10pt; font-style: italic; margin: 6pt 0 2pt; }
.abstract { margin: 8pt 40pt; font-size: 9.5pt; border-left: 3px solid #aaa; padding-left: 8pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; }
th, td { border: 1px solid #bbb; padding: 3pt 6pt; text-align: right; }
th { background: #f0f0f0; font-weight: bold; text-align: center; }
td:first-child { text-align: left; }
.figure { text-align: center; margin: 10pt 0; }
.figure img { max-width: 100%; }
.caption { font-size: 8.5pt; color: #555; margin-top: 4pt; }
.columns { column-count: 2; column-gap: 20pt; }
p { margin: 4pt 0; text-align: justify; }
ul, ol { margin: 4pt 0 4pt 16pt; }
li { margin: 2pt 0; }
.ref-list { font-size: 9pt; }
""")
html.write_pdf(str(OUT), stylesheets=[css])
print(f"Written: {OUT}")
